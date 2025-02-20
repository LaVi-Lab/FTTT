from datasets import load_dataset

from .humaneval_dataset import HumanEvalDataset


class MBPPDataset(HumanEvalDataset):
    template = "{{ question }}"

    @staticmethod
    def get_examples(split, local_dir=None):
        if local_dir is None:
            local_dir = "datasets/mbpp"

        examples = []
        for item in load_dataset(local_dir)[split]:
            examples.append(
                dict(
                    question="You are an expert Python programmer, and here is your task:\\n{text}\\nYour code should pass the following tests:\\n{prompt_tests}".format(text=item["text"], prompt_tests="\n".join(item["test_list"])),
                    answer=dict(
                        prompt="",  # we expect the model itself can generate the complete program
                        test="\ndef check():\n    {}\n".format("\n    ".join(item["test_list"] + item["challenge_test_list"])),
                        entry_point="",
                    ),
                    solution=item["code"],
                    id=item["task_id"],
                )
            )

        return examples
