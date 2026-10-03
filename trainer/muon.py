"""Muon optimizer: orthogonalized momentum for 2-D hidden weights.

Reference: Jordan et al., "Muon: MomentUm Orthogonalized by Newton-Schulz".
Update = Newton-Schulz orthogonalized nesterov momentum; scale by
sqrt(max(1, out/in)). Pair with AdamW for embeddings, norms, and heads.
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T from the SVD of G via Newton-Schulz (quintic)."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov, ns_steps=ns_steps,
                                            weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nest = group["lr"], group["momentum"], group["nesterov"]
            ns_steps, wd = group["ns_steps"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(g)
                buf = state["m"]
                buf.lerp_(g, 1.0 - mom)
                u = g.lerp(buf, mom) if nest else buf
                # Orthogonalize in matrix space; flatten >2-D defensively.
                u2 = u.reshape(u.size(0), -1) if u.ndim > 2 else u
                v = zeropower_via_newtonschulz5(u2, steps=ns_steps)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1.0 - lr * wd)
                p.add_(v.to(p.dtype), alpha=-lr * scale)


def split_params_for_muon(model: torch.nn.Module):
    """(muon_params, adamw_params): Muon gets 2-D hidden weights, AdamW the rest."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "emb" not in name:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
