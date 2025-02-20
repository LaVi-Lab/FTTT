import argparse
import torch
import os
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, set_seed
from functools import partial

from editable_model import EditableModel
from trainer import Trainer
from generator import Generator
from data import get_dataset_cls
from helpers import print_rank_0
from evaluator import Evaluator
from models.modeling_llama import LlamaForCausalLM
from models.modeling_mistral import MistralForCausalLM


GEN_DATA_DIR = os.path.join(os.path.dirname(__file__), "cache")


def add_common_arguments(parser: argparse.ArgumentParser):
    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MBPP", "MATH500", "GSM8K", "HumanEval"],
        required=True,
        help="The dataset name.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="The local data path.",
    )
    parser.add_argument(
        "--max_length",
        type=str,
        default=4096,
        help="The max length of training data.",
    )

    # Inference-time optimization params
    parser.add_argument(
        "--feedback_str",
        type=str,
        default="<|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect.<|eot_id|>",
        help="The verbal feedback for incorrect solutions.",
    )
    parser.add_argument(
        "--feedback_type",
        type=str,
        default="gt",
        choices=["rm", "gt"],
        help="The source of feedback, i.e., reward model (rm) or ground-truth (gt).",
    )

    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Huggingface model name or local path.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use BFloat16."
    )

    # Generation params
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature in [0, 1].",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-P in (0, 1].",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="The number of max new tokens.",
    )

    # Training params
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The training batch size per device.",
    )


def add_generation_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_version",
        type=str,
        required=True,
        help="The version of the generated data.",
    )
    parser.add_argument(
        "--max_train_size",
        type=int,
        default=None,
        help="The max training dataset size.",
    )
    # Generation params
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of generated samples per question.",
    )
    parser.add_argument(
        "--repeat_generate",
        type=int,
        default=1,
        help="Repeat the generation for each batch N times.",
    )
    parser.add_argument(
        "--reflect_str",
        type=str,
        default="<|start_header_id|>user<|end_header_id|>\n\nYour answer is incorrect. Please carefully check the solution and summarize all mistakes in short. Do NOT provide the corrected solution. Do NOT say \"my solution\".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        help="The instruction to elicit self-reflection.",
    )


def add_training_arguments(parser: argparse.ArgumentParser):
    # Data params
    parser.add_argument(
        "--data_version",
        type=str,
        required=True,
        help="The version of the generated data.",
    )
    parser.add_argument(
        "--correct_trial_strategy",
        type=str,
        choices=["exclude", "include", "ground_truth"],
        default="exclude",
        help="Whether to exclude, include successful trials or treat them as the ground truth when generating training data.",
    )
    
    # Model params
    parser.add_argument(
        "--mlp_class",
        type=str,
        default="IDMLP",
        choices=["LoRA"],
        help="The layer type.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Use one network to process activation and gradient at once.",
    )  # default is true
    parser.add_argument(
        "--one_sided",
        action="store_true",
        help="Only process the activation or gradient using a network, whichever is smaller.",
    )
    parser.add_argument(
        "--x_only",
        action="store_true",
        help="Only process the activation using a network.",
    )
    parser.add_argument(
        "--delta_only",
        action="store_true",
        help="Only process the gradient using a network.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalize the predictions.",
    )  # default is true
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=1,
        help="The number of hidden layers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1920,
        help="The hidden layer width.",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="id",
        choices=["id", "xavier"],
        help="The initialization method.",
    )
    parser.add_argument(
        "--act",
        type=str,
        default="relu",
        # choices=["relu"],
        help="The activation function of the hidden layer.",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Whether to share the model across layers.",
    )
    parser.add_argument(
        "--outer_residual",
        action="store_true",
        help="Put model predictions into a residual branch.",
    )

    # Training params
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log statistics every N steps.",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="AdamW",
        help="The torch optimizer name.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.,
        help="The weight decay coefficient.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="The learning rate for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of training epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=10,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau", "cosine_with_min_lr", "warmup_stable_decay"],
        help="The learning rate scheduler name.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="The number of warmup steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="The warmup ratio.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to store the checkpoints.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        required=False,
        default=None,
        help="The checkpoint directory to resume training."
    )

    # Inference-time optimization params
    parser.add_argument(
        "--inner_params",
        type=str,
        nargs="+",
        required=True,
        help="The list of parameter names that will be edited during inference-time optimization.",
    )
    parser.add_argument(
        "--edit_lr",
        type=float,
        default=1e-4,
        help="The inference-time optimization learning rate.",
    )


def add_eval_arguments(parser: argparse.ArgumentParser):
    eval_parser.add_argument(
        "--skip_cases_path",
        type=str,
        default=None,
        help="The path that stores the ids of skipped test cases.",
    )
    eval_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to store the checkpoints.",
    )
    eval_parser.add_argument(
        "--num_trials",
        type=int,
        default=2,
        help="The number of trials. The minimum is 2 as we need at least one sample to start.",
    )


def data_collator(features, feedback_type="gt"):
    # Assume everything is padded
    if features[0]["reward"] is not None:
        if feedback_type == "gt":
            rewards = [float(feature["judge"]) for feature in features]
        else:
            rewards = [feature["reward"] for feature in features]
        rewards = torch.tensor(rewards)
    else:
        rewards = None
    return dict(
        ids=[feature["id"] for feature in features],
        input_ids=torch.stack([feature["input_ids"] for feature in features]),
        attention_mask=torch.stack([feature["attention_mask"] for feature in features]),
        rewards=rewards,
        labels=torch.stack([feature["label_ids"] for feature in features]) if features[0]["label_ids"] is not None else None,
        padding_mask=torch.stack([feature["padding_mask"] for feature in features]) if features[0]["padding_mask"] is not None else None,
        lengths=torch.tensor([feature["length"] for feature in features]),
        chats=[feature["chat"] for feature in features],
        answers=[feature["answer"] for feature in features],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parent_parser = argparse.ArgumentParser()
    add_common_arguments(parent_parser)
    subparsers = parser.add_subparsers(
        dest="command", help="We provide two sub-command: `gen`, `train` and `eval`."
    )
    gen_parser = subparsers.add_parser(
        "gen",
        parents=[parent_parser],
        add_help=False,
        help="Generate training dataset.",
    )
    add_generation_arguments(gen_parser)
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        add_help=False,
        help="Train the model.",
    )
    add_training_arguments(train_parser)
    eval_parser = subparsers.add_parser(
        "eval",
        parents=[parent_parser],
        add_help=False,
        help="Evaluate the model.",
    )
    add_eval_arguments(eval_parser)
    args = parser.parse_args()
    print_rank_0(args)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        # padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate training data
    if args.command == "gen":
        data_path = os.path.join(GEN_DATA_DIR, args.data_version + ".json")
        raw_train_dataset = get_dataset_cls(args)(
            tokenizer=tokenizer,
            split="train",
            max_length=args.max_length,
            max_train_size=args.max_train_size,
            feedback_buffer=len(tokenizer.encode(args.feedback_str, add_special_tokens=False)),
            trials=None,
            local_dir=args.data_path,
        )

        raw_train_dataloader = DataLoader(
            raw_train_dataset,
            shuffle=True,
            collate_fn=partial(data_collator, feedback_type=args.feedback_type),
            batch_size=args.batch_size,
        )
        generator = Generator(args)
        generator.start()
        try:
            generated_train_data = generator.sample(raw_train_dataloader)
        finally:
            generator.close()

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as fout:
            json.dump(generated_train_data, fout)

    # Train the model
    elif args.command == "train":
        data_path = os.path.join(GEN_DATA_DIR, args.data_version + ".json")
        with open(data_path, "r", encoding="utf-8") as fin:
            generated_train_data = json.load(fin)
            
        if torch.cuda.device_count() > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group("gloo", rank=local_rank, world_size=torch.cuda.device_count())

        if "llama" in args.model_name_or_path.lower():
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if args.fp16 else None,
            )
        elif "mistral" in args.model_name_or_path.lower():
            model = MistralForCausalLM.from_pretrained(
                args.model_name_or_path,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if args.fp16 else None,
            )
        else:
            raise ValueError(f"Unknown base model for {args.model_name_or_path}")

        model = EditableModel(model, args)
        print_rank_0(model)
        if torch.cuda.device_count() > 1:
            model = DistributedDataParallel(
                module=model.to(local_rank),
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model = model.cuda()
        
        if args.dataset in ["MBPP", "HumanEval"]:
            generation_start_token = "```python\n"
            generation_end_token = "\n```"
        else:
            generation_start_token, generation_end_token = "", ""
        train_dataset = get_dataset_cls(args)(
            tokenizer=tokenizer,
            split="train",
            max_length=args.max_length,
            max_train_size=None,
            feedback_buffer=len(tokenizer.encode(args.feedback_str, add_special_tokens=False)),
            trials=generated_train_data,
            local_dir=args.data_path,
            correct_trial_strategy=args.correct_trial_strategy,
            generation_start_token=generation_start_token,
            generation_end_token=generation_end_token + tokenizer.eos_token,
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=None if torch.cuda.device_count() > 1 else True,
            collate_fn=partial(data_collator, feedback_type=args.feedback_type),
            batch_size=args.batch_size,
            sampler=DistributedSampler(train_dataset) if torch.cuda.device_count() > 1 else None,
        )

        trainer = Trainer(args, model, tokenizer, len(train_dataloader) * args.num_train_epochs)
        trainer.train(train_dataloader)

        if torch.cuda.device_count() > 1:
            dist.destroy_process_group()
    
    # Evaluate the model
    elif args.command == "eval":
        with open(os.path.join(args.output_dir, "config.json"), "r", encoding="utf-8") as fin:
            config = json.load(fin)
            for k, v in config.items():
                args.__dict__[k] = v

        dataset_cls = get_dataset_cls(args)

        eval_dataset = dataset_cls(
            tokenizer=tokenizer,
            split="test",
            max_length=args.max_length,
            max_train_size=None,
            feedback_buffer=len(tokenizer.encode(args.feedback_str, add_special_tokens=False)),
            trials=None,
            local_dir=args.data_path,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=partial(data_collator, feedback_type=args.feedback_type),
            batch_size=args.batch_size,
        )

        evaluator = Evaluator(args)
        evaluator.start()
        try:
            eval_results = evaluator.run(eval_dataloader)
        finally:
            evaluator.close()

        print_rank_0(f"***** Running results *****")
        print_rank_0(f"  Num. Correct = {eval_results['num_correct']}")
        print_rank_0(f"  Precision    = {eval_results['precision']}")
        print_rank_0(f"  Trial Dist.  = {eval_results['trial_dist']}")

    else:
        raise ValueError(f"Unknown command: {args.command}")
