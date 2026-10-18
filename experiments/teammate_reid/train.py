"""Trainable appearance heads on frozen PRTReID features.

The PRTReID backbone and its pooling heads stay frozen; features are
precomputed once (`features.py`). Exactly these layers train:

- ``proj_global``: one Linear(256 -> 128) on the frozen global embedding. It can
  only re-weight what the runtime embedding already carries.
- ``stripe_head``: per stripe LayerNorm + Linear(1920 -> 64) on the four
  foreground-weighted backbone stripes, a LayerNorm + Linear(256 -> 64) on the
  global embedding, ReLU, then Linear(320 -> 128). It can draw on channels the
  global pooling discards, region by region.

Loss: batch-hard triplet whose negatives are restricted to the *same video and
same team* (teammates, the case the kit cannot separate), plus a cosine-softmax
identity loss whose softmax runs only over the identities of the crop's own
video (identities are never compared across videos). Batches are built from
one (video, team) group at a time, so every negative available is a teammate.
Labels come only from mined, reviewed number-vouched fragments of the training
matches; nothing from a test match and nothing appearance-derived enters.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ProjGlobal(nn.Module):
    def __init__(self, out: int = 128):
        super().__init__()
        self.proj = nn.Linear(256, out)

    def forward(self, stripes, glob):
        return F.normalize(self.proj(glob), dim=1)


class StripeHead(nn.Module):
    def __init__(self, n_stripes: int = 4, stripe_dim: int = 1920, per: int = 64,
                 out: int = 128, dropout: float = 0.2, stripe_drop: float = 0.15):
        super().__init__()
        self.stripes = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(stripe_dim), nn.Dropout(dropout), nn.Linear(stripe_dim, per))
            for _ in range(n_stripes)])
        self.glob = nn.Sequential(nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, per))
        self.out = nn.Linear(per * (n_stripes + 1), out)
        self.stripe_drop = stripe_drop

    def forward(self, stripes, glob, drop: tuple = ()):
        parts = [f(stripes[:, i]) for i, f in enumerate(self.stripes)] + [self.glob(glob)]
        for i in drop:                       # evaluation-time ablation of one region
            parts[i] = torch.zeros_like(parts[i])
        if self.training and self.stripe_drop:
            keep = (torch.rand(len(parts), device=glob.device) >= self.stripe_drop).float()
            parts = [p * k for p, k in zip(parts, keep)]
        return F.normalize(self.out(torch.relu(torch.cat(parts, 1))), dim=1)


HEADS = {"proj_global": ProjGlobal, "stripe_head": StripeHead}


def group_index(labels: list) -> dict:
    """(video, team) -> identity -> crop rows."""
    groups: dict = {}
    for row, (video, team, identity) in enumerate(labels):
        groups.setdefault((video, team), {}).setdefault(identity, []).append(row)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def batch_hard_triplet(z, ident, group, margin: float = 0.3):
    """Hardest positive and hardest same-group (teammate) negative per anchor."""
    d = 1 - z @ z.T
    same = ident[:, None] == ident[None, :]
    teammates = (group[:, None] == group[None, :]) & ~same
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (same & ~eye)
    valid = pos.any(1) & teammates.any(1)
    if not valid.any():
        return z.sum() * 0
    hardest_pos = torch.where(pos, d, torch.full_like(d, -1)).max(1).values
    hardest_neg = torch.where(teammates, d, torch.full_like(d, 9)).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()


def train_head(kind: str, stripes: np.ndarray, glob: np.ndarray, labels: list, *,
               steps: int = 400, ids_per_batch: int = 8, per_id: int = 4, groups_per_batch: int = 2,
               lr: float = 1e-3, weight_decay: float = 5e-4, scale: float = 16.0,
               ce_weight: float = 0.5, seed: int = 0, device: str = "cuda") -> tuple[nn.Module, dict]:
    """Train one head on rows with labels (video, team, identity)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups = group_index(labels)
    if not groups:
        raise ValueError("no (video, team) group with two identities to train on")
    identities = sorted({(v, i) for v, _, i in labels})
    class_of = {key: n for n, key in enumerate(identities)}
    video_of_class = torch.tensor([sorted({v for v, _ in identities}).index(v) for v, _ in identities],
                                  device=device)
    videos = sorted({v for v, _ in identities})
    group_keys = sorted(groups)
    weights = np.array([sum(len(r) for r in groups[k].values()) for k in group_keys], float)
    weights /= weights.sum()

    head = HEADS[kind]().to(device)
    classifier = nn.Parameter(torch.randn(len(identities), 128, device=device) * 0.01)
    optim = torch.optim.AdamW(list(head.parameters()) + [classifier], lr=lr, weight_decay=weight_decay)
    S = torch.from_numpy(stripes.astype(np.float32)).to(device)
    G = torch.from_numpy(glob.astype(np.float32)).to(device)
    group_id = {k: n for n, k in enumerate(group_keys)}
    log = []
    head.train()
    for step in range(steps):
        rows, ident, grp = [], [], []
        for k in rng.choice(len(group_keys), size=groups_per_batch, replace=True, p=weights):
            key = group_keys[k]
            members = list(groups[key])
            for identity in rng.permutation(members)[:ids_per_batch]:
                pool = groups[key][identity]
                rows += list(rng.choice(pool, size=per_id, replace=len(pool) < per_id))
                ident += [class_of[(key[0], identity)]] * per_id
                grp += [group_id[key]] * per_id
        rows_t = torch.tensor(rows, device=device)
        ident_t = torch.tensor(ident, device=device)
        grp_t = torch.tensor(grp, device=device)
        z = head(S[rows_t], G[rows_t])
        loss_tri = batch_hard_triplet(z, ident_t, grp_t)
        logits = scale * z @ F.normalize(classifier, dim=1).T
        # softmax only over the identities of each crop's own video
        own_video = video_of_class[ident_t]
        logits = logits.masked_fill(video_of_class[None, :] != own_video[:, None], -1e4)
        loss_ce = F.cross_entropy(logits, ident_t)
        loss = loss_tri + ce_weight * loss_ce
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0 or step == steps - 1:
            log.append({"step": step, "triplet": float(loss_tri.detach()), "ce": float(loss_ce.detach())})
    head.eval()
    info = {"kind": kind, "trainable_params": sum(p.numel() for p in head.parameters()),
            "identities": len(identities), "videos": videos, "groups": len(group_keys),
            "rows": len(labels), "steps": steps, "seed": seed, "log": log}
    return head, info


@torch.inference_mode()
def embed(head: nn.Module, stripes: np.ndarray, glob: np.ndarray, device: str = "cuda",
          drop: tuple = ()) -> np.ndarray:
    S = torch.from_numpy(stripes.astype(np.float32)).to(device)
    G = torch.from_numpy(glob.astype(np.float32)).to(device)
    if drop and isinstance(head, StripeHead):
        return head(S, G, drop=drop).cpu().numpy()
    return head(S, G).cpu().numpy()
