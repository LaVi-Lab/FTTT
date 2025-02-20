from utils import (
    multiple_gpus_inference,
    get_base_model_and_tokenizer,
    DataArguments,
    ModelArguments,
    InferenceArguments,
    TrainingArguments,
    get_dataset_cls,
)
import queue
import traceback
import transformers
import torch
import math
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

        # Perform optimization for each test instance
        correct = False
        for step in range(
            1,
            math.ceil(inference_args.num_trials / inference_args.num_return_sequences)
            + 1,
        ):
            try:
                # Generate a trial
                with torch.no_grad():
                    # https://github.com/meta-llama/llama/issues/325#issuecomment-1588455611
                    outputs = model.generate(
                        instance["input_ids"].unsqueeze(0).to(model.device),
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
                        num_return_sequences=inference_args.num_return_sequences,
                    )
                    responses = outputs[..., instance["input_ids"].shape[-1] :].detach()
                # Check trial
                completions = tokenizer.batch_decode(
                    responses, skip_special_tokens=True
                )
                # gt = tokenizer.decode(instance["label_ids"], skip_special_tokens=True)
                should_stop = False
                for completion in completions:
                    correct = dataset_cls.is_correct(completion, instance["answer"])
                    if correct:
                        should_stop = True
                        break
                if should_stop:
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
                        num_trials=step * inference_args.num_return_sequences,
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
