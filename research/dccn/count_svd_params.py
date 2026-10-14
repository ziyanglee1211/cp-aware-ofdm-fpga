"""Count parameters of the SVD low-rank control models (paper A8 note)."""

import os
import torch

from config import OfdmConfig
from eval_svd_lowrank import SVDLowRankDCCN, CKPT

for rank in (32, 64):
    tag = "b1_shortcp_pdpfix_seed1"
    basic_sd = torch.load(os.path.join(CKPT, f"basic_{tag}.pt"), map_location="cpu", weights_only=True)
    dccn_sd = torch.load(os.path.join(CKPT, f"dccn_{tag}.pt"), map_location="cpu", weights_only=True)
    cfg = OfdmConfig(nbits=1, long_cp=False)
    model = SVDLowRankDCCN(cfg, basic_sd, dccn_sd, rank)
    total = sum(p.numel() for p in model.parameters())
    per_mod = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        per_mod[name] = n
    print(f"rank {rank}: total real params {total}")
    for k, v in per_mod.items():
        print(f"   {k}: {v}")
    del model
