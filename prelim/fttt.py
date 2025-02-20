import queue
import traceback
import transformers
import torch
import math
import os
import json
from multiprocessing import Queue
from peft import LoraConfig, get_peft_model

from utils import (
    multiple_gpus_inference,
    get_base_model_and_tokenizer,
    DataArguments,
    ModelArguments,
    InferenceArguments,
    TrainingArguments,
    LoraArguments,
    get_dataset_cls,
)


def fttt(
    model_arch,
    peft_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    instance,
    training_args,
    inference_args,
    rank,
    num_training_steps,
    dataset_cls,
):
    correct = False
    num_trials = 0
    # We double num_training_steps in case of generating incomplete solutions
    for step in range(1, num_training_steps * 2 + 2):
        if num_trials >= num_training_steps:
            break
        # Generate a trial
        with torch.no_grad():
            outputs = peft_model.generate(
                instance["input_ids"].unsqueeze(0).to(peft_model.device),
                do_sample=inference_args.temperature > 0,
                temperature=(
                    inference_args.temperature
                    if inference_args.temperature > 0
                    else None
                ),
                top_p=(
                    inference_args.top_p if inference_args.temperature > 0 else None
                ),
                max_new_tokens=inference_args.max_new_tokens,
                num_return_sequences=inference_args.num_return_sequences,
                use_cache=True,
                return_dict_in_generate=True,
            )
            response = outputs.sequences[
                ..., instance["input_ids"].shape[-1] :
            ].detach()
            past_key_values = outputs.past_key_values

        # Check trial
        completions = tokenizer.batch_decode(response, skip_special_tokens=False)
        should_stop = False
        for completion in completions:
            correct = dataset_cls.is_correct(completion, instance["answer"])
            if correct:
                should_stop = True
                break
        if should_stop:
            break
        # if rank == 0:
        #     print("============ completion ================", flush=True)
        #     print(completions[0], flush=True)

        if correct is None:
            # just try again if the generated output is too long
            continue
        else:
            num_trials += 1
            feedback = "Your answer is incorrect."
        if model_arch == "llama":
            feedback = f"<|start_header_id|>user<|end_header_id|>\n\n{feedback}<|eot_id|>"
            feedback = "<|eot_id|>" + feedback
        elif model_arch == "mistral":
            feedback = f"[INST] {feedback}[/INST]"
            feedback = "</s>" + feedback
        # if rank == 0:
        #     print("============ feedback ================", flush=True)
        #     print(feedback, flush=True)
        input_ids = tokenizer.encode(feedback, add_special_tokens=False, return_tensors="pt").to(
            peft_model.device
        )
        for _ in range(training_args.num_optim_step_per_sample):
            outputs = peft_model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return correct, num_trials


def self_reflect_norm(
    model_arch,
    peft_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    instance,
    training_args,
    inference_args,
    rank,
    num_training_steps,
    dataset_cls,
):
    coeff = 1
    if model_arch == "llama":
        feedback = "<|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect."
    elif model_arch == "mistral":
        feedback = "[INST] Your answer is incorrect."
    feedback_tokens = tokenizer(feedback, add_special_tokens=False, return_tensors="pt")["input_ids"]

    correct = False
    num_trials = 0
    # We double num_training_steps in case of generating incomplete solutions
    for step in range(1, num_training_steps * 2 + 2):
        if num_trials >= num_training_steps:
            break
        # Generate a trial
        with torch.no_grad():
            outputs = peft_model.generate(
                instance["input_ids"].unsqueeze(0).to(peft_model.device),
                do_sample=inference_args.temperature > 0,
                temperature=(
                    inference_args.temperature
                    if inference_args.temperature > 0
                    else None
                ),
                top_p=(
                    inference_args.top_p if inference_args.temperature > 0 else None
                ),
                max_new_tokens=inference_args.max_new_tokens,
                num_return_sequences=inference_args.num_return_sequences,
                use_cache=True,
                return_dict_in_generate=True,
            )
            response = outputs.sequences[
                ..., instance["input_ids"].shape[-1] :
            ].detach()
            past_key_values = outputs.past_key_values

        # Check trial
        completions = tokenizer.batch_decode(response, skip_special_tokens=False)
        should_stop = False
        for completion in completions:
            correct = dataset_cls.is_correct(completion, instance["answer"])
            if correct:
                should_stop = True
                break
            # num_trials += 1
        if should_stop:
            break
        # if rank == 0:
        #     print("============ completion ================", flush=True)
        #     print(completions[0], flush=True)

        if correct is None:
            # just try again if the generated output is too long
            continue
        if model_arch == "llama":
            if not completions[0].endswith("<|eot_id|>"):
                continue
            instruction = "Your answer is incorrect. Please carefully check the solution and summarize all mistakes in short. Do NOT provide the corrected solution. Do NOT say \"my solution\"."
            instruction = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif model_arch == "mistral":
            if not completions[0].endswith("</s>"):
                continue
            instruction = "Your answer is incorrect. Carefully check the solution step-by-step and list all mistakes in short. MUST NOT provide the correct answer. Your response MUST be in the third person tone."
            instruction = f"[INST] {instruction} [/INST]"
        instruction_tokens = tokenizer.encode(instruction, return_tensors="pt", add_special_tokens=False).to(peft_model.device)

        # Use the raw model to generate feedback
        with torch.no_grad():
            with peft_model.disable_adapter():
                _outputs = peft_model.generate(
                    torch.cat([outputs.sequences, instruction_tokens], dim=-1),
                    do_sample=inference_args.temperature > 0,
                    temperature=(
                        inference_args.temperature
                        if inference_args.temperature > 0
                        else None
                    ),
                    top_p=(
                        inference_args.top_p if inference_args.temperature > 0 else None
                    ),
                    max_new_tokens=inference_args.max_new_tokens,
                    num_return_sequences=1,
                    # use_cache=True,
                    return_dict_in_generate=True,
                )
        input_ids = _outputs.sequences[..., outputs.sequences.shape[-1] + instruction_tokens.shape[-1] :]
        feedback = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
        if model_arch == "llama":
            if not feedback.endswith("<|eot_id|>"):
                # just try again if the generated output is too long
                continue
            feedback = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect. {feedback}"
        elif model_arch == "mistral":
            if not feedback.endswith("</s>"):
                # just try again if the generated output is too long
                continue
            feedback = f"</s>[INST] Your answer is incorrect. {feedback[:-len("</s>")]} [/INST]"
        # if rank == 0:
        #     print("============ feedback ================", flush=True)
        #     print(feedback, flush=True)
        
        # Train on generated feedback
        input_ids = tokenizer.encode(feedback, add_special_tokens=False, return_tensors="pt").to(peft_model.device)
        for _ in range(training_args.num_optim_step_per_sample):
            outputs = peft_model(
                input_ids=input_ids,
                # labels=labels,
                labels=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Compute loss
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, peft_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            losses = loss_fct(shift_logits, shift_labels)
            loss = losses[:feedback_tokens.shape[-1]].mean() + coeff * losses[feedback_tokens.shape[-1]:].mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        num_trials += 1
    return correct, num_trials


def single_gpu_inference(
    input_queue: Queue,
    output_queue: Queue,
    rank,
    model_args,
    training_args,
    inference_args,
    lora_args,
    data_args,
):
    torch.cuda.set_device(f"cuda:{rank}")
    dataset_cls = get_dataset_cls(data_args)

    model, tokenizer = get_base_model_and_tokenizer(
        rank, model_args, training_args, lora_args
    )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        # layers_to_transform=[30, 31],
        # layers_pattern="layers",
    )

    while True:
        try:
            idx, instance = input_queue.get_nowait()
        except queue.Empty:
            continue

        # Reset everything
        transformers.set_seed(training_args.seed)
        # Model
        peft_model = get_peft_model(model, lora_config)
        # peft_model.config.use_cache = True
        peft_model.train()

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in peft_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in peft_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if training_args.optim == "adamw_torch":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=training_args.learning_rate
            )
        elif training_args.optim == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters, lr=training_args.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer {training_args.optim}")

        # Scheduler and math around the number of training steps.
        num_training_steps = math.ceil(
            inference_args.num_trials / inference_args.num_return_sequences
        )
        lr_scheduler = transformers.get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )

        # Perform optimization for each test instance
        assert training_args.method is not None, "--method must be set."
        try:
            correct, num_trials = eval(training_args.method)(
                model_args.model_arch,
                peft_model,
                tokenizer,
                optimizer,
                lr_scheduler,
                instance,
                training_args,
                inference_args,
                rank,
                num_training_steps,
                dataset_cls,
            )
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
                        num_trials=num_trials,
                    )
                )
                break
            except queue.Full:
                continue


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            InferenceArguments,
            LoraArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        inference_args,
        lora_args,
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
        lora_args,
        single_gpu_inference,
        excluded_cases,
    )
