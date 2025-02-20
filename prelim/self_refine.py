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

        try:
            correct = False

            with torch.no_grad():
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
                )
                generate_ids = outputs[0][instance["input_ids"].shape[-1] :].detach()
            completion = tokenizer.decode(generate_ids, skip_special_tokens=False)

            for step in range(1, int(inference_args.num_trials) + 1):
                # https://github.com/madaan/self-refine/blob/9a206d41e5d2d0c241bb441f41eeadb945afaa55/src/gsm/feedback.py#L19
                if model_args.model_arch == "llama":
                    feedback = "<|start_header_id|>user<|end_header_id|>\n\nThere is an error in the solution above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the solution, and check if everything looks good.<|eot_id|>"
                    if not completion.endswith("<|eot_id|>"):
                        feedback = "<|eot_id|>" + feedback
                elif model_args.model_arch == "mistral":
                    feedback = "[INST] There is an error in the solution above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the solution, and check if everything looks good. [/INST]"
                    if not completion.endswith("</s>"):
                        feedback = "</s>" + feedback
                feedback_tokens = tokenizer.encode(feedback, add_special_tokens=False, return_tensors="pt")
                input_ids = torch.cat([instance["input_ids"].unsqueeze(0), generate_ids.unsqueeze(0).cpu(), feedback_tokens], dim=-1)

                with torch.no_grad():
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
                    generate_ids = outputs[0][input_ids.shape[-1] :].detach()
                completion = tokenizer.decode(generate_ids, skip_special_tokens=False)
                
                # if rank == 0:
                #     print(tokenizer.decode(outputs[0], skip_special_tokens=False), flush=True)

                # Check trial
                if dataset_cls.is_correct(completion, instance["answer"]):
                    correct = True
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

        # if rank == 0:
        #     print(tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=False), flush=True)

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
