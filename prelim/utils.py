import json
import typing
import torch
import transformers
import multiprocessing
import collections
from tqdm import tqdm
from multiprocessing import Queue, Process
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: typing.Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_arch: str = field(default=None, metadata={"help": "The model architecture, llama or mistral."})


@dataclass
class DataArguments:
    dataset: str = field(
        default="MATH500", metadata={"help": "The dataset name, MATH500 or GSM8K for mathematical reasoning and MBPP or HumanEval for code completion."}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimizer name, adamw_torch or sgd."},
    )
    num_optim_step_per_sample: int = field(
        default=1,
        metadata={"help": "The number of optimization step for each failed trial."},
    )
    method: str = field(
        default=None, metadata={"help": "The test-time training method."}
    )


@dataclass
class InferenceArguments:
    num_trials: int = field(
        default=1, metadata={"help": "The number of trials for repeated sampling."}
    )
    temperature: float = field(
        default=0.6, metadata={"help": "The temperature in [0, 1]."}
    )
    top_p: float = field(default=0.95, metadata={"help": "Top-P in (0, 1]."})
    max_new_tokens: int = field(
        default=2048, metadata={"help": "The number of max new tokens."}
    )
    num_return_sequences: int = field(
        default=1, metadata={"help": "The number of samples per step."}
    )


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def get_dataset_cls(data_args):
    if data_args.dataset == "GSM8K":
        from gsm8k_dataset import GSM8KDataset

        return GSM8KDataset
    elif data_args.dataset == "MATH500":
        from math500_dataset import MATH500Dataset

        return MATH500Dataset
    elif data_args.dataset == "HumanEval":
        from humaneval_dataset import HumanEvalDataset

        return HumanEvalDataset
    elif data_args.dataset == "MBPP":
        from mbpp_dataset import MBPPDataset

        return MBPPDataset
    else:
        raise ValueError(f"Unknown dataset name {data_args.dataset}")


def get_base_model_and_tokenizer(rank, model_args, training_args, lora_args):
    torch.cuda.set_device(f"cuda:{rank}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="sdpa",
    ).to(f"cuda:{rank}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def multiple_gpus_inference(
    model_args,
    data_args,
    training_args,
    inference_args,
    lora_args,
    single_gpu_inference,
    excluded_cases=[],
    save_correct_cases=False,
):
    multiprocessing.set_start_method("spawn")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    dataset_cls = get_dataset_cls(data_args)
    eval_examples = dataset_cls.get_examples("test", data_args.data_path)
    eval_dataset = dataset_cls(tokenizer, eval_examples)

    input_queue = Queue()
    output_queue = Queue()

    process_pool = []
    for rank in range(torch.cuda.device_count()):
        p = Process(
            target=single_gpu_inference,
            args=[
                input_queue,
                output_queue,
                rank,
                model_args,
                training_args,
                inference_args,
                lora_args,
                data_args,
            ],
            daemon=True,
        )
        p.start()
        process_pool.append(p)

    test_num = 0
    for instance in eval_dataset:
        if instance["id"] in excluded_cases:
            continue
        input_queue.put((instance["id"], instance))
        test_num += 1

    unsolved = 0
    counter = 0
    num_correct = 0
    num_trials = 0
    correct_cases = []
    trial_count = collections.Counter()
    progress_bar = tqdm(desc="Running", total=test_num)
    while True:
        if counter == test_num:
            break

        output = output_queue.get()
        num_correct += output["correct"] if output["correct"] is not None else 0
        num_trials += output["num_trials"]
        trial_count[output["num_trials"]] += 1
        unsolved += 1 if output["correct"] is None else 0
        counter += 1
        if output["correct"]:
            correct_cases.append(output["idx"])

        if counter % 10 == 0:
            print(
                {
                    "precision": num_correct / counter,
                    "avg_trials": num_trials / counter,
                },
                flush=True,
            )

        progress_bar.update()

    if save_correct_cases:
        with open(
            f"{model_args.model_arch}_{data_args.dataset}_correct_cases.json", "w", encoding="utf-8"
        ) as fout:
            fout.write(json.dumps(correct_cases))

    print("***** Running results *****")
    print(f"  Num. Correct = {num_correct}")
    print(f"  Precision    = {num_correct / test_num}")
    print(f"  Num. Trials  = {num_trials}")
    print(f"  Avg. Trials  = {num_trials / test_num}")
    print(f"  Num. Unsolved = {unsolved}")
    print(f"  Trial Dist.  = {trial_count.most_common()}")

    for p in process_pool:
        p.terminate()
