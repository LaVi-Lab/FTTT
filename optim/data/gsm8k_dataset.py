import json
import os
import re

from .base_dataset import BaseDataset


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


class GSM8KDataset(BaseDataset):
    template = 'Given the following problem, reason and give a final answer to the problem.\nProblem: {{ question }}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'

    @staticmethod
    def is_correct(model_completion, gt):
        ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"

        gt_answer = extract_answer(gt)
        if "The final answer is " not in model_completion:
            pred = INVALID_ANS
        else:
            cand = model_completion.split("The final answer is ")[-1].strip()

            match = ANS_RE.search(cand)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                pred = match_str.rstrip(".")
            else:
                pred = INVALID_ANS
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
            solution = ex["answer"].split("\n#### ")[0] + f"\nThe final answer is {extract_answer(ex['answer'])}."
            ex.update(solution=solution)

        return examples
