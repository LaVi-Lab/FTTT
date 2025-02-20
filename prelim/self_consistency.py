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
import os
import json
from collections import Counter


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
        predicted_answers = Counter()
        pred2raw = {}
        first_occur = {}
        all_completions = []
        for step in range(1, int(inference_args.num_trials) + 1):
            try:
                with torch.no_grad():
                    # https://github.com/meta-llama/llama/issues/325#issuecomment-1588455611
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
                    )
                    response = outputs[0][input_ids.shape[-1] :].detach()
                completion = tokenizer.decode(response, skip_special_tokens=True)
                all_completions.append(completion)
                pred = dataset_cls.extract_answer(completion)
                predicted_answers.update({pred: 1})
                pred2raw[pred] = completion
                if pred not in first_occur:
                    first_occur[pred] = step
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

        # Check trial
        # if rank == 0:
        #     print(predicted_answers.most_common(), flush=True)
        top2_preds = predicted_answers.most_common(2)
        if top2_preds[0][0] is None and len(top2_preds) > 1:
            pred = top2_preds[1][0]
        else:
            pred = top2_preds[0][0]
        completion = pred2raw[pred]
        correct = dataset_cls.is_correct(completion, instance["answer"])
        # if rank == 0:
        #     # print(top2_preds, flush=True)
        #     print("========== extracted ===========")
        #     print(pred, flush=True)
        #     print("========== raw ===========")
        #     print(completion, flush=True)

        # import fasteners

        # path = f"outputs/{model_args.model_arch}_{data_args.dataset}_self-consistency{inference_args.num_trials}_seed{training_args.seed}.jsonl"
        # with fasteners.InterProcessLock(path):
        #     with open(path, "a", encoding="utf-8") as fout:
        #         fout.write(
        #             json.dumps({"id": instance["id"], "completions": all_completions})
        #             + "\n"
        #         )

        while True:
            try:
                output_queue.put_nowait(
                    dict(
                        idx=idx,
                        correct=correct,
                        num_trials=first_occur[pred],
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
