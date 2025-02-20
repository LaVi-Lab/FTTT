import math
import torch
from torch import nn


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes = None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError("cfg.combine cannot be used with one-sided MEND variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = eval(cfg.mlp_class)
        print(f"Building Gradient Transform with MLP class {MlpClass}", flush=True)

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def combined_net():
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x

        if cfg.combine:
            self.mlp = combined_net()
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)
                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)

            if self.k < 2:
                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)


class LoRA(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = "id",
        act: str = "id",
        rank: int = None,
        n_modes: int = None,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(indim, rank, bias=False)
        if act.startswith("dropout"):
            dropout = float(act.split(":")[-1])
            self.dropout = nn.Dropout(p=dropout)
        elif act == "id":
            self.dropout = nn.Identity()
        else:
            raise ValueError(f"LoRA does not support --act '{act}'")
        self.up = nn.Linear(rank, outdim)

        if init == "id":
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))

            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            raise ValueError(f"Unrecognized initialization function '{init}'")

        print(f"Building ({init}) LoRA: {[indim, rank, outdim]})", flush=True)
    
    def forward(self, x, mode=None):
        residual = x
        return residual + self.up(self.dropout(self.down(x)))
