#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import logging
import math
import os
from functools import partial
from tqdm import tqdm
from collections import Counter

import torch
from transformers import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
)

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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
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
        "--max_new_tokens",
        type=int,
        default=4096,
        help="The number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.,
        help="The sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The nuclus sampling hyperparameter.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The sampled generation number.",
    )
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=1,
        help="Repeat the sampling N times.",
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

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16 if args.fp16 else None,
    )

    if "Prefix" in args.output_dir or "Adapter" in args.output_dir or "ReFT" in args.output_dir:
        adapters.init(model)
        adapter_name = model.load_adapter(args.output_dir)
        model.set_active_adapters(adapter_name)
        if args.fp16:
            model.bfloat16()
        logger.info(f"Num. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif "BitFit" in args.output_dir or "FT" in args.output_dir:
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=torch.bfloat16 if args.fp16 else None,
        )
    elif "LoRA" in args.output_dir or "IA3" in args.output_dir or "LNTuning" in args.output_dir:
        model.load_adapter(args.output_dir)
    else:
        raise ValueError(f"Unknown PEFT method from {args.output_dir}.")
    logger.info(model)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
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

    if torch.cuda.is_available():
        model.cuda()

    # greedy decode and evaluate accuracy on the test set
    correct_ids = []
    n_total = 0
    trial_count = Counter()
    for idx, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=math.ceil(len(eval_dataset) / args.per_device_eval_batch_size)):
        if batch is None:
            print(f"{idx} is skipped", flush=True)
            continue
    
        completions = []
        for i in range(args.num_repeat):
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=args.num_return_sequences if args.temperature > 0 else 1,
                )

            output_ids = outputs.sequences[
                ..., batch["input_ids"].shape[-1] :
            ].detach()

            completions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        completions = [completions]
        # print(completions, flush=True)

        for _id, gt, _completions in zip(batch["ids"], batch["answers"], completions):
            n_total += 1
            for step, completion in enumerate(_completions):
                correct = dataset_cls.is_correct(completion, gt)
                if correct:
                    correct_ids.append(_id)
                    trial_count.update({step: 1})
                    break
            else:
                print(_id, _completions, flush=True)

    logger.info(f"***** Running evaluation *****")
    logger.info(f"  Total        = {n_total}")
    logger.info(f"  Num. Correct = {len(correct_ids)}")
    logger.info(f"  Precision    = {len(correct_ids) / n_total}")
    logger.info(f"  Trial Dist.  = {trial_count.most_common()}")


if __name__ == "__main__":
    main()
