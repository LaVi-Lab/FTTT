#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import logging
import math
import os
import sys
from functools import partial
from tqdm import tqdm

import torch
from transformers import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

from peft import LoraConfig, get_peft_model, IA3Config, LNTuningConfig
import adapters


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use bfloat16 for faster training.",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=1,
        help="Save the model every N epcohs.",
    )

    parser.add_argument(
        "--peft_type",
        type=str,
        required=True,
        choices=["FT", "BitFit", "Prefix", "LoRA", "Adapter", "IA3", "LNTuning"],
        help="The parameter-efficient fine-tuning method.",
    )
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="The list of module names that LoRA will be applied.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.,
        help="The dropout ratio of LoRA.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices=['none', 'all', 'lora_only'],
        help="The bias type for LoRA.",
    )
    parser.add_argument(
        "--lora_layers_to_transform",
        type=int,
        nargs="+",
        default=None,
        help="The list of layer ids that LoRA will be applied.",
    )
    # Prefix arguments
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=30,
        help="The trainable prefix length.",
    )
    # Adapter arguments
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=16,
        help="The reduction factor for the bottleneck layer."
    )
    parser.add_argument(
        "--leave_out",
        type=str,
        default=None,
        help="The layers where Adapter is not added.",
    )
    # (Lo)ReFT arguments
    parser.add_argument(
        "--reft_layers",
        type=str,
        default="all",
        help="The layers where ReFT is applied.",
    )
    parser.add_argument(
        "--reft_prefix_positions",
        type=int,
        default=3,
        help="The number of prefix tokens that ReFT is applied.",
    )
    parser.add_argument(
        "--reft_suffix_positions",
        type=int,
        default=0,
        help="The number of suffix tokens that ReFT is applied.",
    )
    parser.add_argument(
        "--reft_r",
        type=int,
        default=4,
        help="The rank of LoReFT.",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.,
        help="The warmup steps ratio.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="The maximum training data length.",
    )
    parser.add_argument(
        "--skip_cases_path",
        type=str,
        default=None,
        help="The path that stores the ids of skipped test cases.",
    )
    args = parser.parse_args()

    return args


def data_collator(features, skip_cases=[]):
    # Assume everything is padded
    selected_features = [feature for feature in features if feature["id"] not in skip_cases]
    if len(selected_features) == 0:
        return None
    else:
        return dict(
            ids=[feature["id"] for feature in selected_features],
            input_ids=torch.stack([feature["input_ids"] for feature in selected_features]),
            attention_mask=torch.stack([feature["attention_mask"] for feature in selected_features]),
            labels=torch.stack([feature["label_ids"] for feature in selected_features]) if selected_features[0]["label_ids"] is not None else None,
            answers=[feature["answer"] for feature in selected_features],
        )


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.fp16 else None,
    )

    if args.peft_type == "BitFit":
        for n, p in model.named_parameters():
            if "bias" not in n:
                p.requires_grad = False
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif args.peft_type == "Prefix":
        from adapters import PrefixTuningConfig

        config = PrefixTuningConfig(flat=False, prefix_length=args.prefix_length)
        adapters.init(model)
        model.add_adapter(args.peft_type, config=config)
        model.train_adapter(args.peft_type)
        if args.fp16:
            model.bfloat16()
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif args.peft_type == "Adapter":
        from adapters import BnConfig

        config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=args.reduction_factor, non_linearity="relu", leave_out=args.leave_out.split(",") if args.leave_out is not None else [])
        adapters.init(model)
        model.add_adapter(args.peft_type, config=config)
        model.train_adapter(args.peft_type)
        if args.fp16:
            model.bfloat16()
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif args.peft_type == "IA3":
        ia3_config = IA3Config(
            task_type="CAUSAL_LM",
            target_modules=None,
            feedforward_modules=None,
        )
        model = get_peft_model(model, ia3_config)
        model.print_trainable_parameters()
    elif args.peft_type == "LNTuning":
        ln_config = LNTuningConfig(
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, ln_config)
        if args.fp16:
            model.bfloat16()
        model.print_trainable_parameters()
    elif args.peft_type == "ReFT":
        from adapters import ReftConfig

        config = ReftConfig(
            layers=args.reft_layers.split(",") if args.reft_layers != "all" else args.reft_layers, prefix_positions=args.reft_prefix_positions, suffix_positions=args.reft_suffix_positions, r=args.reft_r, orthogonality=True
        )  # equivalent to LoreftConfig()
        adapters.init(model)
        model.add_adapter(args.peft_type, config=config)
        model.train_adapter(args.peft_type)
        # TODO: use bfloat16 when setting orthogonality=True
        model.adapter_to(args.peft_type, dtype=torch.float32)
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif args.peft_type == "LoRA":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            layers_to_transform=args.lora_layers_to_transform,
            layers_pattern="layers",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        assert args.peft_type == "FT", f"Unknown `peft_type` '{args.peft_type}'"
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name in ["MBPP", "HumanEval"]:
        generation_start_token = "```python\n"
        generation_end_token = "\n```"
    else:
        generation_start_token, generation_end_token = "", ""

    if args.dataset_name == "GSM8K":
        from data import GSM8KDataset
        dataset_cls = GSM8KDataset
    elif args.dataset_name == "MATH500":
        from data import MATH500Dataset
        dataset_cls = MATH500Dataset
    elif args.dataset_name == "HumanEval":
        from data import HumanEvalDataset
        dataset_cls = HumanEvalDataset
    elif args.dataset_name == "MBPP":
        from data import MBPPDataset
        dataset_cls = MBPPDataset
    else:
        raise ValueError(f"Unknown dataset name {args.dataset_name}")

    train_dataset = dataset_cls(
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_length,
        max_train_size=None,
        feedback_buffer=0,
        trials=None,
        generation_start_token=generation_start_token,
        generation_end_token=generation_end_token + tokenizer.eos_token,
        mode="baseline_train",
    )
    eval_dataset = dataset_cls(
        tokenizer=tokenizer,
        split="test",
        max_length=args.max_length,
        max_train_size=None,
        feedback_buffer=0,
        trials=None,
        mode="baseline_test",
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    skip_cases = []
    if args.skip_cases_path is not None and os.path.exists(args.skip_cases_path):
        with open(args.skip_cases_path, "r", encoding="utf-8") as fin:
            skip_cases.extend(json.load(fin))
    else:
        logger.warning(f"{args.skip_cases_path} does not exist.")
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=partial(data_collator, skip_cases=skip_cases),
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.num_warmup_steps is None:
        args.num_warmup_steps = args.num_train_epochs * num_update_steps_per_epoch * args.warmup_ratio

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )
    if torch.cuda.is_available():
        model.cuda()

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), desc="Training")
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    num_used_batches = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss.backward()
            if num_used_batches != 0 and num_used_batches % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                completed_steps += 1
            num_used_batches += 1

            if completed_steps >= args.max_train_steps:
                break

        if (epoch + 1) % args.save_every_n_epochs == 0:
            output_dir = f"epoch-{epoch + 1}"
            if args.peft_type in ["Prefix", "Adapter", "ReFT"]:
                model.save_adapter(os.path.join(args.output_dir, output_dir), args.peft_type)
            else:
                model.save_pretrained(os.path.join(args.output_dir, output_dir))
        
        logger.info({"epoch": epoch, "step": completed_steps, "train_loss": total_loss.item() / len(train_dataloader), "lr": lr_scheduler.get_lr()})

    output_dir = f"epoch-{args.num_train_epochs}"
    if not os.path.exists(os.path.join(args.output_dir, output_dir)):
        if args.peft_type in ["Prefix", "Adapter"]:
            model.save_adapter(os.path.join(args.output_dir, output_dir), args.peft_type)
        else:
            model.save_pretrained(os.path.join(args.output_dir, output_dir))

    # greedy decode and evaluate accuracy on the test set
    greedy_correct_ids = []
    n_total = 0
    for idx, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        if batch is None:
            # print(f"{idx} is skipped", flush=True)
            continue
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                do_sample=False,
                max_new_tokens=4096,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_ids = outputs.sequences[
            ..., batch["input_ids"].shape[-1] :
        ].detach()

        completions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for _id, gt, completion in zip(batch["ids"], batch["answers"], completions):
            n_total += 1
            correct = dataset_cls.is_correct(completion, gt)
            if correct:
                greedy_correct_ids.append(_id)

    logger.info(f"***** Running evaluation *****")
    logger.info(f"  Total                    = {n_total}")
    logger.info(f"  Num. Correct (Greedy)    = {len(greedy_correct_ids)}")
    logger.info(f"  Precision (Greedy)       = {len(greedy_correct_ids) / n_total}")


if __name__ == "__main__":
    main()
