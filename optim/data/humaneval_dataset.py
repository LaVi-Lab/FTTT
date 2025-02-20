from datasets import load_dataset
import requests
import time
import os
import re

from .base_dataset import BaseDataset


NUM_RETRIES = 3
RETRY_BACKOFF = 3


class HumanEvalDataset(BaseDataset):
    template = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n\n\n{{ question }}\n\n```"

    @staticmethod
    def is_correct(model_completion, gt):
        code = extract_first_code(model_completion)
        if code is None:
            return None

        for i in range(NUM_RETRIES):
            try:
                response = requests.post(
                    os.environ.get("OJ_API"),
                    json={
                        "problem": gt,
                        "completion": code,
                    },
                )
                response.raise_for_status()
                is_correct = response.json()["passed"]
                break
            except:
                print(code, flush=True)
                if i == NUM_RETRIES - 1:
                    raise
                time.sleep(RETRY_BACKOFF**i)

        return is_correct

    @staticmethod
    def get_examples(split, local_dir=None):
        if local_dir is None:
            local_dir = "datasets/openai_humaneval"

        examples = []
        for item in load_dataset(local_dir)[split]:

            examples.append(
                dict(
                    question=item["prompt"],
                    answer=dict(
                        prompt="",  # we expect the model itself can generate the complete program
                        canonical_solution=item["canonical_solution"],
                        test=item["test"],
                        entry_point=item["entry_point"],
                    ),
                    solution=item["canonical_solution"],
                    id=item["task_id"],
                )
            )

        return examples


def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    except MemoryError:
        return False


def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None
