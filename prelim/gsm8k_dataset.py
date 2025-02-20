import json
import os
import re
import torch as th
from jinja2 import Template


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt):
    gt_answer = extract_answer(gt)
    pred = extract_answer(model_completion)
    assert gt_answer != INVALID_ANS
    return pred == gt_answer


class GSM8KDataset(th.utils.data.Dataset):
    template = 'Given the following problem, reason and give a final answer to the problem.\nProblem: {{ question }}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'

    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.chats = [
            [
                {
                    "role": "user",
                    "content": Template(self.template).render(question=ex["question"]),
                },
            ]
            for ex in self.examples
        ]
        self.qns = [
            tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            )
            for chat in self.chats
        ]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False, add_special_tokens=False)
        self.ans = tokenizer(self.ans, padding=False, add_special_tokens=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return dict(
            input_ids=th.tensor(self.qns["input_ids"][idx]),
            label_ids=th.tensor(self.ans["input_ids"][idx]),
            question=self.examples[idx]["question"],
            answer=self.examples[idx]["answer"],
            id=self.examples[idx]["id"],
            chat=self.chats[idx],
        )

    @staticmethod
    def extract_answer(model_completion):
        ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")

        if "The final answer is " not in model_completion:
            # pred = INVALID_ANS
            return None
        else:
            cand = model_completion.split("The final answer is ")[-1].strip()

            match = ANS_RE.search(cand)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                pred = match_str.rstrip(".")
            else:
                # pred = INVALID_ANS
                return None
        return pred 


    @staticmethod
    def is_correct(model_completion, gt):
        ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"

        gt_answer = extract_answer(gt)
        if "The final answer is " not in model_completion:
            return None
        else:
            cand = model_completion.split("The final answer is ")[-1].strip()

            match = ANS_RE.search(cand)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                pred = match_str.rstrip(".")
            else:
                return None
        assert gt_answer != INVALID_ANS
        return pred == gt_answer

    @staticmethod
    def get_examples(split, local_dir=None):
        if local_dir is None:
            local_dir = "datasets/grade-school-math/grade_school_math"

        path = os.path.join(local_dir, "data/", f"{split}.jsonl")
        examples = read_jsonl(path)

        for idx, ex in enumerate(examples):
            ex.update(id=idx)

        print(f"{len(examples)} {split} examples")
        return examples
