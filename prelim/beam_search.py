from utils import (
    multiple_gpus_inference,
    get_base_model_and_tokenizer,
    DataArguments,
    ModelArguments,
    InferenceArguments,
    get_dataset_cls,
    TrainingArguments,
)
import queue
import traceback
import transformers
import torch
import os
import json


def single_gpu_inference(
    input_queue,
    output_queue,
    rank,
    model_args,
    training_args,
    inference_args,
    lora_args,
    data_args,
):
    model, tokenizer = get_base_model_and_tokenizer(
        rank, model_args, training_args, lora_args
    )
    dataset_cls = get_dataset_cls(data_args)

    while True:
        try:
            idx, instance = input_queue.get_nowait()
        except queue.Empty:
            continue
        
        transformers.set_seed(training_args.seed)

        input_ids = instance["input_ids"].unsqueeze(0)
        try:
            with torch.no_grad():
                # TODO: for stochastic beam search, we can simply run `num_trials` times and took the one with the highest score as the result for each run
                num_trials = inference_args.num_trials
                while num_trials > 0:
                    try:
                        outputs = model.generate(
                            input_ids.to(model.device),
                            do_sample=inference_args.temperature > 0,
                            temperature=(
                                inference_args.temperature
                                if inference_args.temperature > 0
                                else None
                            ),
                            top_p=(
                                inference_args.top_p
                                if inference_args.temperature > 0
                                else None
                            ),
                            max_new_tokens=inference_args.max_new_tokens,
                            num_beams=num_trials,
                            num_return_sequences=num_trials,
                        )
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        num_trials = num_trials // 2
                        print(f"[RANK {rank}]: OOM, reduce beam size to {num_trials}", flush=True)
                if num_trials == 0:
                    raise RuntimeError("beam size reach 0 and still OOM, consider reducing max_new_tokens")
                responses = outputs[:, input_ids.shape[-1] :].detach()
            # Check trial
            completions = tokenizer.batch_decode(responses, skip_special_tokens=True)
            # if rank == 0:
            #     print(completions, flush=True)
            for step, completion in enumerate(completions):
                correct = dataset_cls.is_correct(completion, instance["answer"])
                if correct:
                    break
        except Exception as e:
            traceback.print_exc()
            print(f"[Rank {rank}]: ", e, flush=True)
            while True:
                try:
                    input_queue.put_nowait((idx, instance))
                    break
                except queue.Full:
                    continue
            return

        while True:
            try:
                output_queue.put_nowait(
                    dict(
                        idx=idx,
                        correct=correct,
                        num_trials=step,
                    )
                )
                break
            except queue.Full:
                continue


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, InferenceArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        inference_args,
    ) = parser.parse_args_into_dataclasses()

    if os.path.exists(f"metadata/{model_args.model_arch}_{data_args.dataset}_correct_cases.json"):
        with open(f"metadata/{model_args.model_arch}_{data_args.dataset}_correct_cases.json", "r", encoding="utf-8") as fin:
            excluded_cases = json.load(fin)
    else:
        print("Running on all test data...", flush=True)

    multiple_gpus_inference(
        model_args,
        data_args,
        training_args,
        inference_args,
        None,
        single_gpu_inference,
        excluded_cases,
    )
