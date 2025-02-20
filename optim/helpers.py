import requests
import os


def print_rank_0(*args, **kwargs):
    flush = kwargs.pop("flush", True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("[INFO]:", *args, **kwargs, flush=flush)
    

def score_completions(dataset_cls, chats, batch_completions, gts, feedback_type):
    rewards = []
    judges = []
    for chat, completions, gt in zip(chats, batch_completions, gts):
        _rewards = []
        _judges = []
        for completion in completions:
            judge = dataset_cls.is_correct(completion, gt)
            if judge is None:  # Fail to extract answers
                judge = False
            if feedback_type == "rm":
                while True:
                    response = requests.post(
                        url=os.environ.get("RM_API"),
                        json={
                            "chat": chat
                            + [{"role": "assistant", "content": completion}]
                        },
                    )
                    try:
                        response.raise_for_status()
                        reward = response.json()
                        break
                    except Exception as e:
                        print(response.text, flush=True)
            elif feedback_type == "gt":
                reward = float(judge)
            else:
                raise NotImplementedError
            _rewards.append(reward)
            _judges.append(judge)
        rewards.append(_rewards)
        judges.append(_judges)
    return rewards, judges
