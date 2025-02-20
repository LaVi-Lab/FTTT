import torch
import transformers
import math
import time
import datetime
import os
import json
import re

from tqdm import tqdm
import torch.distributed as dist

from helpers import print_rank_0
from data import get_dataset_cls


class Trainer:
    def __init__(self, config, model, tokenizer, num_training_steps) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.rank = int(os.environ.get("LOCAL_RANK", 0))

        if config.resume_checkpoint is not None:
            match = re.search(r"(?:^|/|~)/?epoch-(\d+)", config.resume_checkpoint)
            if match:
                self.starting_epoch = int(match.group(1)) - 1
            else:
                raise ValueError(f"Cannot parse the epoch from {config.resume_checkpoint}")
        else:
            self.starting_epoch = -1

        self.dataset_cls = get_dataset_cls(config)

        self.OptimizerClass = getattr(torch.optim, config.opt)
        # print_rank_0(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            module = self.model.module
        else:
            module = self.model
        no_decay = ["bias", "layer_norm.weight", "edit_lrs", "mode_shift", "mode_scale"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in module.named_outer_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in module.named_outer_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.opt = self.OptimizerClass(optimizer_grouped_parameters, lr=config.lr)
        # TODO: MEND has a different lr for learnable lr

        warmup_steps = (
            config.warmup_steps if config.warmup_steps > 0 else math.ceil(num_training_steps * config.warmup_ratio)
        )
        self.lr_scheduler = transformers.get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            # last_epoch=self.starting_epoch,
        )

        if config.resume_checkpoint is not None:
            print_rank_0(f"Resuming training from {config.resume_checkpoint}")

            state_dict = torch.load(os.path.join(config.resume_checkpoint, "model.ckpt"))
            self.model.load_state_dict(state_dict)

            opt_state = torch.load(os.path.join(config.resume_checkpoint, "state.ckpt"))
            self.opt.load_state_dict(opt_state["optimizer"])
            self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])
    
    def get_config(self):
        return dict(
            inner_params=self.config.inner_params,
            edit_lr=self.config.edit_lr,
            fp16=self.config.fp16,
            combine=self.config.combine,
            one_sided=self.config.one_sided,
            x_only=self.config.x_only,
            delta_only=self.config.delta_only,
            n_hidden=self.config.n_hidden,
            init=self.config.init,
            act=self.config.act,
            rank=self.config.rank,
            norm=self.config.norm,
            mlp_class=self.config.mlp_class,
            outer_residual=self.config.outer_residual,
            shared=self.config.shared,
        )
    
    def get_device(self):
        if torch.cuda.device_count() == 1:
            device = "cuda"
        else:
            device = "cpu"
        return device

    def prepare_inputs_from_feedbacks(self, input_ids, attention_mask, rewards):
        feedback_tokens = self.tokenizer.encode(
            self.config.feedback_str, return_tensors="pt", add_special_tokens=False
        ).to(input_ids.device).repeat(input_ids.shape[0], 1)
        new_input_ids = torch.cat([input_ids, feedback_tokens], dim=1)
        new_attention_mask = torch.cat([attention_mask, torch.ones_like(feedback_tokens)], dim=1)
        return new_input_ids, new_attention_mask

    def train(self, train_dataloader):
        device = self.get_device()
        start_time = time.time()
        self.model.train()

        past_steps = 0
        losses = []
        for epoch in range(self.config.num_train_epochs):
            if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            pbar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch + 1}", total=len(train_dataloader)) if self.rank == 0 else None
            for step, batch in enumerate(train_dataloader):
                if epoch <= self.starting_epoch:
                    if pbar is not None:
                        pbar.update(1)
                    continue

                # Edit the model based on feedbacks
                input_ids, attention_mask = self.prepare_inputs_from_feedbacks(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    rewards=batch["rewards"].to(device),
                )
                named_delta, loss = self.model(
                    input_ids=input_ids.to(device),  # left-padded
                    attention_mask=attention_mask.to(device),
                    rewards=batch["rewards"].to(device),
                    labels=batch["labels"].to(device),  # right-padded
                    padding_mask=batch["padding_mask"].to(device),
                    lengths=batch["lengths"].to(device),
                )

                losses.append(loss.detach().float().cpu())

                loss.backward()

                nan, inf = False, False
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    model = self.model.module
                else:
                    model = self.model
                for p in model.outer_parameters():
                    if p.grad is not None:
                        nan |= p.grad.isnan().any().item()
                        inf |= p.grad.isinf().any().item()

                if nan or inf:
                    print_rank_0(f"Skipping the current batch because inf: {inf} nan: {nan}")
                    self.model.zero_grad()

                if step > 0 and step % self.config.gradient_accumulation == 0:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                    past_steps += 1

                    if past_steps > 0 and past_steps % self.config.logging_steps == 0:
                        print_rank_0(
                            dict(
                                epoch=epoch + 1,
                                batch=step,
                                loss=torch.mean(torch.stack(losses)).item(),
                                lr=self.lr_scheduler.get_lr(),
                                time=str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            )
                        )
                        losses = []
                
                if pbar is not None:
                    pbar.update(1)

            if self.rank == 0 and epoch > self.starting_epoch:
                ckpt_dir = os.path.join(self.config.output_dir, f"epoch-{epoch + 1}")
                print_rank_0(f"Saving to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(), os.path.join(ckpt_dir, "model.ckpt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "optimizer": self.opt.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    os.path.join(ckpt_dir, "state.ckpt")
                )
                with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as fout:
                    json.dump(self.get_config(), fout, indent=4, ensure_ascii=False)
            
            if torch.cuda.device_count() > 1:
                dist.barrier()
