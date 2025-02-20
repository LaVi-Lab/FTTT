from typing import Dict, List
from pydantic import BaseModel

import transformers
import argparse
import torch
import fastapi
import uvicorn


app = fastapi.FastAPI()


class ScoreChatRequest(BaseModel):
    chat: List[Dict[str, str]]


if __name__ == "__main__":
    # python rm_api.py
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-Math-RM-72B",
        help="The reward model HF name or local path.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="The API host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6003,
        help="The API port.",
    )
    args = parser.parse_args()

    model = transformers.AutoModel.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    @app.post("/")
    def predict(req: ScoreChatRequest):
        conversation_str = tokenizer.apply_chat_template(
            req.chat, tokenize=False, add_generation_prompt=False
        )

        input_ids = tokenizer.encode(
            conversation_str, return_tensors="pt", add_special_tokens=False
        ).to(model.device)[..., -model.config.max_position_embeddings :]

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        reward = outputs[0].detach().cpu().float().numpy().item()
        return reward

    uvicorn.run(app, host=args.host, port=args.port)
