from .gsm8k_dataset import GSM8KDataset
from .humaneval_dataset import HumanEvalDataset
from .math500_dataset import MATH500Dataset
from .mbpp_dataset import MBPPDataset


def get_dataset_cls(data_args):
    if data_args.dataset == "GSM8K":
        return GSM8KDataset
    elif data_args.dataset == "MATH500":
        return MATH500Dataset
    elif data_args.dataset == "HumanEval":
        return HumanEvalDataset
    elif data_args.dataset == "MBPP":
        return MBPPDataset
    else:
        raise ValueError(f"Unknown dataset name {data_args.dataset}")
