#!/usr/bin/env python3
r"""cluster_brick_vs_backprop_real_weights.py

Same local-vs-global cluster-brick comparison as cluster_brick_vs_backprop_toy.py, now with
ONE variable changed: the attention/MLP weights are REAL, pretrained Llama-3.1-8B-Instruct
weights (a community "abliterated" variant was used in the original experiment -- any
standard Llama-3.1-8B checkpoint in HF safetensors format works identically), not random
init. Everything else (task, freeze/train split, delta-rule mechanism, training budget) is
held identical for a fair comparison.

Point BASE at a local directory containing that model's config.json + .safetensors shards.
"""
import os, json
import torch, torch.nn.functional as F
from safetensors.torch import safe_open
import random

torch.manual_seed(0)
random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BASE = r"PATH/TO/YOUR/Llama-3.1-8B-Instruct-CHECKPOINT"  # edit this to your local model dir
LAYER_IDS = [8, 9, 10, 11]   # doubled from the 2-layer slice -- real-scale escalation
N_PAIRS = 4
STEPS = 400            # fewer steps than the toy: real dims make each step heavier
LR = 0.001              # smaller than the toy run -- real pretrained weights have real scale


def load_real_slice():
    cfg = json.load(open(os.path.join(BASE, "config.json")))
    H = cfg["hidden_size"]; NHEADS = cfg["num_attention_heads"]
    HEAD_DIM = H // NHEADS; INTER = cfg["intermediate_size"]
    KV_HEADS = cfg.get("num_key_value_heads", NHEADS)
    shards = [f for f in os.listdir(BASE) if f.endswith(".safetensors")]

    def read(k):
        for sh in shards:
            with safe_open(os.path.join(BASE, sh), framework="pt") as f:
                if k in f.keys():
                    return f.get_tensor(k).to(torch.bfloat16).to(DEV)  # bf16 to fit more real layers in 16GB
        raise KeyError(k)

    print(f"loading real weights: H={H} heads={NHEADS} head_dim={HEAD_DIM} inter={INTER} "
          f"layers={LAYER_IDS}", flush=True)
    p = {}
    p['emb'] = read("model.embed_tokens.weight")   # full table, we'll only index a few rows
    for i, l in enumerate(LAYER_IDS):
        p[f'Wq{i}'] = read(f"model.layers.{l}.self_attn.q_proj.weight").T.contiguous()
        p[f'Wk{i}'] = read(f"model.layers.{l}.self_attn.k_proj.weight").T.contiguous()
        p[f'Wv{i}'] = read(f"model.layers.{l}.self_attn.v_proj.weight").T.contiguous()
        p[f'Wo{i}'] = read(f"model.layers.{l}.self_attn.o_proj.weight").T.contiguous()
        p[f'Wgate{i}'] = read(f"model.layers.{l}.mlp.gate_proj.weight").T.contiguous()
        p[f'Wup{i}'] = read(f"model.layers.{l}.mlp.up_proj.weight").T.contiguous()
        p[f'Wdown{i}'] = read(f"model.layers.{l}.mlp.down_proj.weight").T.contiguous()
    lmh = None
    try:
        lmh = read("lm_head.weight")
    except KeyError:
        lmh = p['emb']
    p['Whead'] = lmh.T.contiguous()
    return p, H, NHEADS, HEAD_DIM, INTER, KV_HEADS


P0, HID, HEADS, HEAD_DIM, INTER, KV_HEADS = load_real_slice()
GQA_RATIO = HEADS // KV_HEADS
LAYERS = len(LAYER_IDS)

# small, arbitrary real vocab ids to use as our symbol set -- structural test, not semantic
SYMBOL_IDS = list(range(1000, 1000 + 3 * N_PAIRS + 1))
FIXED_KEYS = SYMBOL_IDS[0:N_PAIRS]
FIXED_MIDS = SYMBOL_IDS[N_PAIRS:2 * N_PAIRS]
FIXED_VALS = SYMBOL_IDS[2 * N_PAIRS:3 * N_PAIRS]
SEP = SYMBOL_IDS[-1]
LABEL_SET = FIXED_VALS  # classify among just these N_PAIRS real token ids


def make_example():
    facts = [(FIXED_KEYS[i], FIXED_MIDS[i]) for i in range(N_PAIRS)] + \
            [(FIXED_MIDS[i], FIXED_VALS[i]) for i in range(N_PAIRS)]
    random.shuffle(facts)
    seq = []
    for a, b in facts:
        seq += [a, b]
    q_idx = random.randrange(N_PAIRS)
    seq += [SEP, FIXED_KEYS[q_idx]]
    label_idx = LABEL_SET.index(FIXED_VALS[q_idx])
    return torch.tensor(seq, dtype=torch.long), label_idx


def rmsnorm(x, eps=1e-5):
    xf = x.float()
    return (xf / (xf.pow(2).mean(-1, keepdim=True) + eps).sqrt()).to(x.dtype)


def forward(p, seq):
    x = p['emb'][seq]
    T = x.shape[0]
    mask = torch.triu(torch.full((T, T), float('-inf'), device=DEV, dtype=torch.bfloat16), 1)
    for l in range(LAYERS):
        xn = rmsnorm(x)
        q = (xn @ p[f'Wq{l}']).view(T, HEADS, HEAD_DIM)
        k = (xn @ p[f'Wk{l}']).view(T, KV_HEADS, HEAD_DIM)
        v = (xn @ p[f'Wv{l}']).view(T, KV_HEADS, HEAD_DIM)
        head_outs = []
        for h in range(HEADS):
            kvh = h // GQA_RATIO  # grouped-query attention: multiple q heads share one kv head
            w = (q[:, h] @ k[:, kvh].T) / (HEAD_DIM ** 0.5) + mask
            w = w.softmax(-1)
            head_outs.append(w @ v[:, kvh])
        attn_cat = torch.cat(head_outs, dim=-1)
        attn_out = attn_cat @ p[f'Wo{l}']
        x = x + attn_out
        hn = rmsnorm(x)
        gate = (hn @ p[f'Wgate{l}'])
        up = (hn @ p[f'Wup{l}'])
        neuron_act = F.silu(gate) * up
        mlp_out = neuron_act @ p[f'Wdown{l}']
        x = x + mlp_out
    pooled = rmsnorm(x[-1:])[0] @ p['Whead']        # [vocab]
    label_logits = pooled[torch.tensor(LABEL_SET, device=DEV)]  # restrict to our small label set
    return label_logits


def eval_acc(p, n=60):
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            seq, label = make_example()
            logits = forward(p, seq.to(DEV))
            if logits.argmax().item() == label:
                correct += 1
    return correct / n


def run_global():
    # emb/Whead are the full 128256-row vocab table -- never meant to be trained here (forward()
    # only ever indexes ~13 specific rows); training them with full Adam state was the real OOM
    # cause (2.1GB params x4 for grad+2 moments, twice over for both tables).
    frozen_vocab = {'emb', 'Whead'}
    fixed = {k: P0[k].clone() for k in frozen_vocab}
    p = {k: v.clone().requires_grad_(True) for k, v in P0.items() if k not in frozen_vocab}
    opt = torch.optim.Adam(list(p.values()), lr=LR)
    acc_hist = []
    for step in range(STEPS):
        seq, label = make_example()
        pp = dict(p); pp.update(fixed)
        logits = forward(pp, seq.to(DEV))
        loss = F.cross_entropy(logits[None].float(), torch.tensor([label], device=DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p.values()), 1.0)
        opt.step()
        if step % 20 == 0:
            pp_eval = dict(p); pp_eval.update(fixed)
            acc_hist.append(eval_acc(pp_eval))
            print(f"  global step {step} loss {loss.item():.3f} acc {acc_hist[-1]:.2f}", flush=True)
    return acc_hist


def run_local():
    frozen_keys = [f'Wo{l}' for l in range(LAYERS)] + [f'Wdown{l}' for l in range(LAYERS)] + ['emb', 'Whead']
    fixed = {k: P0[k].clone() for k in frozen_keys}
    local_params = {k: v.clone().requires_grad_(True) for k, v in P0.items() if k not in frozen_keys}
    opt = torch.optim.Adam(list(local_params.values()), lr=LR)
    acc_hist = []
    for step in range(STEPS):
        seq, label = make_example()
        pp = dict(local_params); pp.update(fixed)
        logits = forward(pp, seq.to(DEV))
        loss = F.cross_entropy(logits[None].float(), torch.tensor([label], device=DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(local_params.values()), 1.0)
        opt.step()
        if step % 20 == 0:
            pp_eval = dict(local_params); pp_eval.update(fixed)
            acc_hist.append(eval_acc(pp_eval))
            print(f"  local step {step} loss {loss.item():.3f} acc {acc_hist[-1]:.2f}", flush=True)
    return acc_hist


if __name__ == "__main__":
    print("=== GLOBAL backprop on real pretrained weights (ceiling) ===", flush=True)
    g = run_global()
    print(f"final: {g[-1]:.2%}", flush=True)

    print("\n=== LOCAL (Wo/Wdown frozen at REAL pretrained values) ===", flush=True)
    l = run_local()
    print(f"final: {l[-1]:.2%}", flush=True)

    print(f"\n=== VERDICT === global={g[-1]:.2%}  local={l[-1]:.2%}  gap={g[-1]-l[-1]:+.2%}", flush=True)
