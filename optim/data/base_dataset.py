import os
import torch
import random
from typing import Literal
from copy import deepcopy
from jinja2 import Template


def print_rank_0(*args, **kwargs):
    flush = kwargs.pop("flush", True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("[INFO]:", *args, **kwargs, flush=flush)


class BaseDataset(torch.utils.data.Dataset):
    template = None

    def __init__(self, tokenizer, split, max_length, max_train_size=None, feedback_buffer=0, trials=None, local_dir=None, correct_trial_strategy: Literal["exclude", "include", "ground_truth"] = "exclude", generation_start_token="", generation_end_token="", mode: Literal["edit", "baseline_train", "baseline_test"] = "edit"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Read raw data
        raw_examples = []
        for ex in self.get_examples(split, local_dir):
            new_ex = deepcopy(ex)
            new_ex["chat"] = [
                {
                    "role": "user",
                    "content": Template(self.template).render(question=ex["question"]),
                },
            ]
            new_ex["question"] = tokenizer.apply_chat_template(
                new_ex["chat"],
                add_generation_prompt=True,
                tokenize=False,
            )
            new_ex["question_tokens"] = tokenizer(new_ex["question"], padding=False, add_special_tokens=False)
            new_ex["solution_tokens"] = tokenizer(generation_start_token + new_ex["solution"] + generation_end_token, padding=False, add_special_tokens=False)
            raw_examples.append(new_ex)
        print_rank_0(f"Example #1 In:\n{tokenizer.batch_decode([raw_examples[0]['question_tokens']['input_ids']], skip_special_tokens=False)[0]}")
        print_rank_0(f"Example #1 Out:\n{tokenizer.batch_decode([raw_examples[0]['solution_tokens']['input_ids']], skip_special_tokens=False)[0]}")
        print_rank_0(f"{len(raw_examples)} {split} examples loaded")
        if max_train_size is not None:
            raw_examples = random.sample(raw_examples, k=min(max_train_size, len(raw_examples)))
        self.max_q_len = max([len(ex["question_tokens"]["input_ids"]) for ex in raw_examples])

        # Attach generated data
        if trials is not None:
            examples = []
            for ex in raw_examples:
                try:
                    for i, (completion, reward, judge) in enumerate(zip(trials[str(ex["id"])]["completions"], trials[str(ex["id"])]["rewards"], trials[str(ex["id"])]["judges"])):
                        if correct_trial_strategy == "exclude" and judge:
                            continue
                        new_ex = deepcopy(ex)
                        new_ex["completion"] = completion
                        new_ex["completion_tokens"] = tokenizer(new_ex["completion"], padding=False)
                        new_ex["reward"] = reward
                        new_ex["judge"] = judge
                        new_ex["id"] = str(ex["id"]) + f"@{i}"
                        if correct_trial_strategy == "ground_truth" and judge:
                            new_ex["solution_tokens"] = new_ex["completion_tokens"]
                        examples.append(new_ex)
                except KeyError:
                    continue
            raw_examples = examples
            print_rank_0(f"{len(raw_examples)} {split} examples augmented with {correct_trial_strategy} strategy for correct trials")

        # Filter lengthy examples
        self.examples = []
        for ex in raw_examples:
            if len(ex["question_tokens"]["input_ids"]) + len(ex["solution_tokens"]["input_ids"]) > max_length:
                continue
            if "completion_tokens" in ex:
                if len(ex["question_tokens"]["input_ids"]) + len(ex["completion_tokens"]["input_ids"]) + feedback_buffer > max_length:
                    continue
            self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        question_tokens = example["question_tokens"]["input_ids"]
        solution_tokens = example["solution_tokens"]["input_ids"]

        if self.mode == "baseline_train":
            return dict(
                id=example["id"],
                input_ids=torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - len(question_tokens) - len(solution_tokens)) + question_tokens + solution_tokens),
                attention_mask=torch.tensor([0] * (self.max_length - len(question_tokens) - len(solution_tokens)) + [1] * (len(question_tokens) + len(solution_tokens))),
                label_ids=torch.tensor([-100] * (self.max_length - len(solution_tokens)) + solution_tokens),
                answer=example["answer"],
            )
        elif self.mode == "baseline_test":
            return dict(
                id=example["id"],
                input_ids=torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - len(question_tokens)) + question_tokens),
                attention_mask=torch.tensor([0] * (self.max_length - len(question_tokens)) + [1] * (len(question_tokens))),
                label_ids=None,
                answer=example["answer"],
            )

        length = len(example["question_tokens"]["input_ids"])
        # For training
        if "completion" in example:
            completion_tokens = example["completion_tokens"]["input_ids"]
            # Left padding
            input_ids = [self.tokenizer.pad_token_id] * (self.max_length - len(question_tokens) - len(completion_tokens)) + question_tokens + completion_tokens
            attention_mask = [0] * (self.max_length - len(question_tokens) - len(completion_tokens)) + [1] * (len(question_tokens) + len(completion_tokens))
            # Right padding
            label_ids = question_tokens + solution_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(question_tokens) - len(solution_tokens))
            padding_mask = [1] * (len(question_tokens) + len(solution_tokens)) + [0] * (self.max_length - len(question_tokens) - len(solution_tokens))

            reward = example["reward"]
            judge = example["judge"]
        # For data generation
        else:
            # Left padding
            input_ids = [self.tokenizer.pad_token_id] * (self.max_q_len - len(question_tokens)) + question_tokens
            attention_mask = [0] * (self.max_q_len - len(question_tokens)) + [1] * len(question_tokens)
            reward = None
            label_ids = None
            padding_mask = None
            judge=None

        return dict(
            id=example["id"],
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
            reward=reward,
            label_ids=torch.tensor(label_ids) if label_ids is not None else None,
            padding_mask=torch.tensor(padding_mask) if padding_mask is not None else None,
            length=length,
            chat=example["chat"],
            answer=example["answer"],
            judge=judge,
        )

    @staticmethod
    def is_correct(model_completion, gt):
        raise NotImplementedError

    @staticmethod
    def get_examples(split, local_dir=None):
        raise NotImplementedError
