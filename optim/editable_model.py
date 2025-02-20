import os
import torch
import transformers
from torch import nn
from collections import defaultdict
from typing import Optional

from modules import GradientTransform
from hooks import hook_model
from helpers import print_rank_0


def _inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


class EditableModel(nn.Module):
    def __init__(self, model, config, edit_lrs=None):
        super().__init__()

        self.base_model = model
        self.config = config

        # Try to not store the gradients of parameters in lower layers
        min_layer_idx = None
        for name in self.config.inner_params:
            for item in name.split("."):
                try:
                    layer_idx = int(item)
                    break
                except:
                    continue
            min_layer_idx = min(float("inf") if min_layer_idx is None else min_layer_idx, layer_idx)
        assert min_layer_idx is not None
        for n, p in self.base_model.named_parameters():
            for item in n.split("."):
                try:
                    layer_idx = int(item)
                    break
                except:
                    continue
            if layer_idx < min_layer_idx:
                p.requires_grad = False

        # Even not to compute the gradient w.r.t activations to save GPU RAM
        setattr(self.base_model.config, "no_grad_layers", min_layer_idx)

        device = self.get_device()
        if edit_lrs is None:
            edit_lrs = nn.Parameter(torch.tensor([config.edit_lr] * len(self.config.inner_params), device=device))
        self.edit_lrs = edit_lrs

        if not hasattr(self.base_model, "handles"):
            hook_model(self.base_model, self.config.inner_params)
            # print_rank_0(f"Hooked {len(self.model.handles)//2} modules")

        if self.config.shared:
            shape_dict = defaultdict(list)
            for n, p in _inner_params(model.named_parameters(), self.config.inner_params):
                shape_dict[self.get_shape(p)].append(n)
            self.shape_dict = shape_dict

            self.optim = nn.ModuleDict({
                str(tuple(s)): GradientTransform(*s, config, len(shape_dict[s]))
                for s in shape_dict.keys()
            }).to(device)
        else:
            self.optim = nn.ModuleDict({
                n.replace(".", "#"): GradientTransform(*self.get_shape(p), config)
                for (n, p) in _inner_params(model.named_parameters(), self.config.inner_params)
            })
        
        print_rank_0(f"Num. Trainable Parameters: {sum(p.numel() for p in self.optim.parameters() if p.requires_grad):,}")

        # if config.fp16:
        #     self.optim.bfloat16()
    
    def get_device(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.device_count() > 0:
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        return device

    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return p.shape if isinstance(self.base_model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.base_model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"base_model.{k}"]
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert len([k for k in res.missing_keys if not k.startswith("base_model.")]) == 0, "Should only have missing keys for model."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self):
        return list(self.optim.parameters()) + [self.edit_lrs]

    def named_outer_parameters(self):
        return list(self.optim.named_parameters()) + [["edit_lrs", self.edit_lrs]]

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def forward(
        self,
        # Edit Model
        input_ids: torch.LongTensor = None,  # left-padded
        attention_mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        # Compute Loss
        labels: Optional[torch.LongTensor] = None,  # right-padded
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.LongTensor] = None,
    ):
        ######################## Edit Model ########################
        names = set([n for n, p in self.base_model.named_parameters()])
        pset = set(self.config.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        _labels = input_ids.clone()
        if attention_mask is not None:
            _labels[~attention_mask.bool()] = -100
        output = self.base_model(
            input_ids=input_ids,  # question + completion + feedback
            attention_mask=attention_mask,
            labels=_labels,  # question + completion + feedback (with pad_token masked for loss calculation)
        )
        # TODO: add reward to loss
        loss = output.loss
        loss.backward()

        # Compute & apply updates
        if self.config.shared:
            param_idx = lambda n, p: self.shape_dict[self.get_shape(p)].index(n)  # noqa: E731
            transformed_factors = {}
            for n, p in _inner_params(self.base_model.named_parameters(), self.config.inner_params):
                u = p.__x__
                # Ensure masked positions do not contribute
                if attention_mask is not None:
                    v = p.__delta__ * attention_mask.unsqueeze(-1).type_as(p.__delta__)
                transformed_factors[n] = self.optim[str(tuple(self.get_shape(p)))](u, v, param_idx(n, p))
        else:
            transformed_factors = {}
            for n, p in _inner_params(self.base_model.named_parameters(), self.config.inner_params):
                u = p.__x__
                # Ensure masked positions do not contribute
                if attention_mask is not None:
                    v = p.__delta__ * attention_mask.unsqueeze(-1).type_as(p.__delta__)
                transformed_factors[n] = self.optim[n.replace(".", "#")](u, v)

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        if isinstance(self.base_model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        mean_grads = {
            n: torch.einsum(f"bi,bj->{targ}", x, delta)
            for n, (x, delta) in transformed_factors.items()
        }  # TODO: bi,bj->bij for batching in a single device

        if self.config.outer_residual:
            for n, p in _inner_params(self.base_model.named_parameters(), self.config.inner_params):
                mean_grads[n] += p.grad

        self.base_model.zero_grad()

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        named_delta = {}
        for lr, (n, g) in zip(self.edit_lrs, mean_grads.items()):
            named_delta[n] = (lr * g).bfloat16() if self.config.fp16 else lr * g

        ######################## Compute Loss ########################
        loss = None
        if labels is not None:
            label_mask = torch.logical_and(
                torch.arange(labels.shape[-1], device=padding_mask.device)[None, :] >= (lengths[:, None]),
                padding_mask,
            )

            _labels = labels.clone()
            _labels[~label_mask.bool()] = -100
            outputs = self.base_model(
                input_ids=labels,  # question + solution
                labels=_labels,  # question + solution
                named_delta=named_delta,
            )

            loss = outputs.loss
            # TODO: regularization for positive feedback: predictions should be zeros for correct solutions

        return named_delta, loss
