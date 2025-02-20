import os
import torch
import queue
import argparse
import traceback
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from tqdm import tqdm

from helpers import score_completions, print_rank_0
from data import get_dataset_cls


def generate_worker(i: int, input_queue: queue.Queue, output_queue: queue.Queue, config: argparse.Namespace):
    os.environ["LOCAL_RANK"] = i

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            # padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.fp16 else None,
        ).to(f"cuda:{i}")

        num_return_sequences = 1
        print_rank_0("`--num_return_sequences` has no effect and is always set to 1.")
    except Exception as e:
        traceback.print_exc()
        print(e, flush=True)
        return

    while True:
        if input_queue.empty():
            continue

        try:
            batch = input_queue.get_nowait()
        except:
            continue

        try:
            completions = []
            reflections = []
            with torch.inference_mode():
                for _ in range(config.repeat_generate):
                    past_key_values = DynamicCache()
                    outputs = model.generate(
                        input_ids=batch["input_ids"].to(model.device),  # left-padded
                        attention_mask=batch["attention_mask"].to(model.device),
                        do_sample=config.temperature > 0,
                        temperature=(
                            config.temperature
                            if config.temperature > 0
                            else None
                        ),
                        top_p=(
                            config.top_p if config.temperature > 0 else None
                        ),
                        max_new_tokens=config.max_new_tokens,
                        num_return_sequences=num_return_sequences,
                        use_cache=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=past_key_values,
                    )

                    generate_ids = outputs.sequences[
                        ..., batch["input_ids"].shape[-1] :
                    ].detach()

                    # skip incomplete outputs
                    if not torch.all(generate_ids[:, -1] == tokenizer.pad_token_id).cpu().item():
                        continue
                    
                    # generate reflection
                    reflect_tokens = tokenizer.encode(config.reflect_str, return_tensors="pt", add_special_tokens=False).repeat(batch["input_ids"].shape[0], 1).to(model.device)
                    _outputs = model.generate(
                        input_ids=torch.cat([outputs.sequences, reflect_tokens], dim=-1),  # left-padded
                        attention_mask=torch.cat([batch["attention_mask"].to(model.device), torch.ones_like(generate_ids), torch.ones_like(reflect_tokens)], dim=-1),
                        do_sample=config.temperature > 0,
                        temperature=(
                            config.temperature
                            if config.temperature > 0
                            else None
                        ),
                        top_p=(
                            config.top_p if config.temperature > 0 else None
                        ),
                        max_new_tokens=config.max_new_tokens,
                        num_return_sequences=num_return_sequences,
                        use_cache=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=past_key_values,
                    )

                    _generate_ids = _outputs.sequences[
                        ..., outputs.sequences.shape[-1] + reflect_tokens.shape[-1] :
                    ].detach()

                    # skip incomplete outputs
                    if not torch.all(_generate_ids[:, -1] == tokenizer.pad_token_id).cpu().item():
                        continue

                    completions.extend(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))  # batch_size x num_return_sequences
                    reflections.extend(tokenizer.batch_decode(_generate_ids, skip_special_tokens=True))

                    # print("================ completion =================", flush=True)
                    # print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True), flush=True)
                    # print("================ reflection =================", flush=True)
                    # print(tokenizer.batch_decode(_generate_ids, skip_special_tokens=True), flush=True)
            completions = [completions[i * num_return_sequences * config.repeat_generate : (i + 1) * num_return_sequences * config.repeat_generate] for i in range(batch["input_ids"].shape[0])]
            reflections = [reflections[i * num_return_sequences * config.repeat_generate : (i + 1) * num_return_sequences * config.repeat_generate] for i in range(batch["input_ids"].shape[0])]
        except Exception as e:
            traceback.print_exc()
            print(f"Rank {i}: {e}", flush=True)

            while True:
                try:
                    input_queue.put_nowait(batch)
                    break
                except:
                    continue
            return

        while True:
            try:
                output_queue.put_nowait(
                    dict(
                        ids=batch["ids"],
                        completions=completions,
                        reflections=reflections,
                        chats=batch["chats"],
                        answers=batch["answers"],
                        status=batch["status"] + 1,
                    )
                )
                break
            except:
                continue

        del batch


class Generator:
    def __init__(self, config) -> None:
        self.config = config

        self.dataset_cls = get_dataset_cls(config)
    
    def start(self):
        context = mp.get_context("spawn")
        self.input_queue = context.Queue()
        self.output_queue = context.Queue()
        self.mp_context = mp.spawn(
            fn=generate_worker,
            args=(
                self.input_queue,
                self.output_queue,
                self.config,
            ),
            nprocs=torch.cuda.device_count(),
            join=False,
            daemon=True,
            start_method="spawn",
        )

    def close(self):
        for p in self.mp_context.processes:
            p.terminate()
        self.input_queue.close()
        self.output_queue.close()

    def sample(self, dataloader):
        for batch in dataloader:
            batch["status"] = 0
            while True:
                try:
                    self.input_queue.put_nowait(batch)
                    break
                except:
                    continue

        gen_data = {}
        for step in tqdm(range(len(dataloader)), desc="Generating Train Data"):
            batch = self.output_queue.get()
            batch_rewards, batch_judges = score_completions(
                dataset_cls=self.dataset_cls,
                chats=batch["chats"],  # batch_size
                batch_completions=batch["completions"],  # [batch_size, num_return_sequences * repeat_generate]
                gts=batch["answers"],  # batch_size
                feedback_type=self.config.feedback_type,
            )  # [batch_size, num_return_sequences * repeat_generate]

            for _id, completions, reflections, rewards, judges in zip(batch["ids"], batch["completions"], batch["reflections"], batch_rewards, batch_judges):
                # assert _id not in gen_data
                gen_data[_id] = dict(
                    completions=[],
                    reflections=[],
                    rewards=[],
                    judges=[],
                )
                for completion, reflection, reward, judge in zip(completions, reflections, rewards, judges):
                    gen_data[_id]["completions"].append(completion)
                    gen_data[_id]["reflections"].append(reflection)
                    gen_data[_id]["rewards"].append(reward)
                    gen_data[_id]["judges"].append(judge)
            
        return gen_data
