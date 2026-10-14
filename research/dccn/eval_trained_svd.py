"""Evaluate checkpoints produced by train_svd_lowrank.py on paired BER."""
import argparse, json, os
import torch
from config import OfdmConfig
from eval_svd_lowrank import SVDLowRankDCCN, DEV, CKPT, evaluate

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--nbits', type=int, default=1)
    ap.add_argument('--cp', default='long', choices=['long', 'short'])
    ap.add_argument('--source-suffix', default='')
    ap.add_argument('--channels', default='EVA,ETU')
    ap.add_argument('--snrs', default='12,20,24,28')
    ap.add_argument('--slots', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--train-seed', type=int, default=1)
    ap.add_argument('--eval-seed', type=int, default=20260913)
    args = ap.parse_args()
    cfg = OfdmConfig(args.nbits, args.cp == 'long')
    tag = f"b{args.nbits}_{args.cp}cp"
    src = args.source_suffix
    b = torch.load(os.path.join(CKPT, f'basic_{tag}{src}.pt'), map_location='cpu', weights_only=True)
    d = torch.load(os.path.join(CKPT, f'dccn_{tag}{src}.pt'), map_location='cpu', weights_only=True)
    m = SVDLowRankDCCN(cfg, b, d, args.rank).to(DEV)
    path = os.path.join(CKPT, f'svd_{tag}_r{args.rank}_seed{args.train_seed}.pt')
    m.load_state_dict(torch.load(path, map_location=DEV, weights_only=True)); m.eval()
    channels = [x.strip() for x in args.channels.split(',') if x.strip()]
    snrs = [float(x) for x in args.snrs.split(',')]
    result = evaluate(m, cfg, channels, snrs, args.slots, args.batch, args.eval_seed)
    out = os.path.join(os.path.dirname(CKPT), 'results', f'svd_retrained_{tag}_r{args.rank}_seed{args.train_seed}.json')
    with open(out, 'w') as f: json.dump({'rank': args.rank, 'channels': channels, 'snrs': snrs, 'slots': args.slots, 'results': result}, f, indent=2)
    print('saved', out)
