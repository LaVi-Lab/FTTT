import os
import torch
import queue
import argparse
import traceback
import json
from collections import Counter
import torch.multiprocessing as mp
from transformers import AutoTokenizer, set_seed

from tqdm import tqdm

from helpers import score_completions
from data import get_dataset_cls
from models.modeling_llama import LlamaForCausalLM
from models.modeling_mistral import MistralForCausalLM
from editable_model import EditableModel


def evaluate_worker(i: int, input_queue: queue.Queue, output_queue: queue.Queue, config: argparse.Namespace):
    os.environ["LOCAL_RANK"] = i
    device = f"cuda:{i}"
    dataset_cls = get_dataset_cls(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        # padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "llama" in config.model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            config.model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if config.fp16 else None,
        )
    elif "mistral" in config.model_name_or_path.lower():
        model = MistralForCausalLM.from_pretrained(
            config.model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if config.fp16 else None,
        )
    else:
        raise ValueError(f"Unknown architecture for {config.model_name_or_path}")

    state_dict = torch.load(os.path.join(config.output_dir, "model.ckpt"))
    model = EditableModel(model, config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    num_trials = config.num_trials

    while True:
        if input_queue.empty():
            continue

        try:
            batch = input_queue.get_nowait()
        except:
            continue

        set_seed(config.seed)

        try:
            named_delta = None
            feedback_tokens = tokenizer.encode(
                config.feedback_str, return_tensors="pt", add_special_tokens=False
            ).to(device).repeat(batch["input_ids"].shape[0], 1)

            for trial in range(num_trials):
                # torch.cuda.empty_cache()

                # Generate trials via greedy decoding
                if trial % 2 == 1:
                    current_named_delta = named_delta
                    do_sample = config.temperature > 0
                else:
                    current_named_delta = None
                    do_sample = config.temperature > 0
                for _ in range(8):
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            max_new_tokens=config.max_new_tokens,
                            do_sample=do_sample,
                            temperature=(
                                config.temperature
                                if config.temperature > 0
                                else None
                            ),
                            top_p=(
                                config.top_p if config.temperature > 0 else None
                            ),
                            return_dict=True,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                            named_delta=current_named_delta,
                        )

                    generate_ids = outputs[..., batch["input_ids"].shape[-1] :]
                    # in case of the generated trial is not complete, try several times until all end with pad token
                    if not do_sample or torch.all(generate_ids[:, -1] == tokenizer.pad_token_id).cpu().item():
                        break

                completions = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                completions = [[completion] for completion in completions]
                # print(completions, flush=True)

                # Evaluate the completions
                rewards, judges = score_completions(
                    dataset_cls=dataset_cls,
                    chats=batch["chats"],  # batch_size
                    batch_completions=completions,  # [batch_size, num_return_sequences]
                    gts=batch["answers"],  # batch_size
                    feedback_type=config.feedback_type,
                )  # [batch_size, num_return_sequences]

                assert len(judges) == 1  # batch_size x num_return_sequences
                if judges[0][0]:
                    # print(batch["ids"], completions, flush=True)
                    break

                # Edit the model based on feedbacks
                if trial < num_trials - 1:  # No need to perform edit in the last trial
                    named_delta, _ = model(
                        input_ids=torch.cat([batch["input_ids"].to(device), generate_ids, feedback_tokens], dim=1),
                        attention_mask=torch.cat(
                            [
                                batch["attention_mask"].to(device), 
                                torch.ones(
                                    batch["attention_mask"].shape[0], 
                                    generate_ids.shape[1] + feedback_tokens.shape[1], 
                                    device=device, 
                                    dtype=batch["attention_mask"].dtype,
                                )
                            ], dim=1),
                        rewards=torch.tensor(rewards).squeeze(-1),
                    )
                
                del outputs
            
            final_judges = [judge[0] for judge in judges]  # Report the last completions
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
                        judges=final_judges,
                        trial=trial if final_judges[0] else config.num_trials,
                    )
                )
                break
            except:
                continue

        del batch


class Evaluator:
    def __init__(self, config) -> None:
        self.config = config

        self.skip_cases = []
        if config.skip_cases_path is not None and os.path.exists(config.skip_cases_path):
            with open(config.skip_cases_path, "r", encoding="utf-8") as fin:
                self.skip_cases.extend(json.load(fin))
    
    def start(self):
        context = mp.get_context("spawn")
        self.input_queue = context.Queue()
        self.output_queue = context.Queue()
        self.mp_context = mp.spawn(
            fn=evaluate_worker,
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

    def run(self, dataloader):
        counter = 0
        for batch in dataloader:
            assert len(batch["ids"]) == 1
            if batch["ids"][0] in self.skip_cases:
                continue
            while True:
                try:
                    self.input_queue.put_nowait(batch)
                    counter += 1
                    break
                except:
                    continue

        eval_results = []
        trial_count = Counter()
        for step in tqdm(range(counter), desc="Evaluating"):
            batch = self.output_queue.get()
            eval_results.extend(batch["judges"])
            trial_count.update({batch["trial"]: 1})
            
        return dict(
            num_correct=sum(eval_results),
            precision=sum(eval_results) / len(eval_results),
            trial_dist=trial_count.most_common(),
        )
