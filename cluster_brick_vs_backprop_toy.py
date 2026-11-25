#!/usr/bin/env python3
r"""prototype_cluster_brick_transformer.py -- 2026-11-24.

Question (Jonathan, 2026-11-24): can brick-style LOCAL learning -- proven so far only on the
fly-connectome reservoir+local-readout substrate (Book 9 Appendix G Part 3: fixed random
reservoir, local delta-rule readout, error 0.296->0.00000, "learns perfectly") and on hand-built
toy nets (prototype_two_brick.py, local_deep_residual.py) -- be applied to a REAL transformer
(Llama) architecture's own structure, not just a custom substrate? "Shrimp eating a llama."

Why a naive hidden-dim split doesn't work (reasoned through, not guessed): the fly connectome is
genuinely SPARSE (~16.8M edges / 138K neurons) -- a local update only ever touches real neighbors.
A transformer's q/k/v/o/gate/up/down projections are DENSE matrices: every output unit depends on
every input unit. An arbitrary slice of the hidden dimension has no real locality to exploit.

The REAL seams that exist without inventing anything: attention HEADS (each head reads its own
slice of q/k/v; heads don't interact until o_proj mixes them) and MLP intermediate NEURONS (each
SwiGLU neuron's output adds linearly into down_proj's sum -- structurally a wide reservoir with a
linear readout). This test treats o_proj and down_proj as the FIXED random "reservoir mixing" (the
same role played by the fly connectome's fixed random wiring) and trains ONLY the per-head q/k/v
and per-neuron-group gate/up weights via a LOCAL DELTA RULE (the exact mechanism already proven in
Book 9 Appendix G Part 3), each cluster never seeing another cluster's gradient. Compared directly
against full end-to-end backprop on the identical block as the ceiling.

Task: tiny synthetic associative recall (same spirit as the proven "favorite color" test and the
fly-reservoir's 8-association test) -- a short token sequence states a few KEY->VALUE pairs, then
asks for one key's value at the end. Small enough to run in seconds on CPU, no GPU needed.
"""
import torch, torch.nn.functional as F
import random

torch.manual_seed(0)
random.seed(0)

VOCAB = 14
HID = 64
HEADS = 4
HEAD_DIM = HID // HEADS
INTER = 128
SEQ_LEN = 12
LAYERS = 2   # stacked blocks -- does local credit survive crossing a layer boundary?
N_PAIRS = 4
LR_GLOBAL = 0.01
LR_LOCAL = 0.01
STEPS = 3000


# HARDER: 2-HOP CHAINED associative memory -- key->mid is one fact, mid->val is a SEPARATE
# fact; the query gives the key, the answer requires chaining key->mid->val. A single layer's
# direct lookup can't shortcut this the way a 1-hop key->val task could -- it genuinely needs
# the two facts composed, a real test of whether credit survives crossing the layer boundary
# rather than depth being along for the ride. Still FIXED associations (matches the proven
# fly-reservoir shape), only the presentation order of the fact-pairs shuffles.
FIXED_KEYS = list(range(1, 1 + N_PAIRS))
FIXED_MIDS = list(range(1 + N_PAIRS, 1 + 2 * N_PAIRS))
FIXED_VALS = list(range(1 + 2 * N_PAIRS, 1 + 3 * N_PAIRS))
random.shuffle(FIXED_MIDS)
random.shuffle(FIXED_VALS)


def make_example():
    facts = [(FIXED_KEYS[i], FIXED_MIDS[i]) for i in range(N_PAIRS)] +             [(FIXED_MIDS[i], FIXED_VALS[i]) for i in range(N_PAIRS)]
    random.shuffle(facts)  # presentation order varies, the chain itself does not
    seq = []
    for a, b in facts:
        seq += [a, b]
    sep = 0
    q_idx = random.randrange(N_PAIRS)
    seq += [sep, FIXED_KEYS[q_idx]]
    label = FIXED_VALS[q_idx]
    return torch.tensor(seq, dtype=torch.long), label


def init_block():
    p = dict(emb=torch.randn(VOCAB, HID) * 0.1, Whead=torch.randn(HID, VOCAB) * 0.1)
    for l in range(LAYERS):
        p[f'Wq{l}'] = torch.randn(HID, HID) * 0.1
        p[f'Wk{l}'] = torch.randn(HID, HID) * 0.1
        p[f'Wv{l}'] = torch.randn(HID, HID) * 0.1
        p[f'Wo{l}'] = torch.randn(HID, HID) * 0.1
        p[f'Wgate{l}'] = torch.randn(HID, INTER) * 0.1
        p[f'Wup{l}'] = torch.randn(HID, INTER) * 0.1
        p[f'Wdown{l}'] = torch.randn(INTER, HID) * 0.1
    return p


def rmsnorm(x, eps=1e-5):
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def forward(p, seq):
    x = p['emb'][seq]
    T = x.shape[0]
    mask = torch.triu(torch.full((T, T), float('-inf')), 1)
    for l in range(LAYERS):
        xn = rmsnorm(x)
        q = (xn @ p[f'Wq{l}']).view(T, HEADS, HEAD_DIM)
        k = (xn @ p[f'Wk{l}']).view(T, HEADS, HEAD_DIM)
        v = (xn @ p[f'Wv{l}']).view(T, HEADS, HEAD_DIM)
        head_outs = []
        for h in range(HEADS):
            w = (q[:, h] @ k[:, h].T) / (HEAD_DIM ** 0.5) + mask
            w = w.softmax(-1)
            head_outs.append(w @ v[:, h])
        attn_cat = torch.cat(head_outs, dim=-1)
        attn_out = attn_cat @ p[f'Wo{l}']
        x = x + attn_out
        hn = rmsnorm(x)
        gate = (hn @ p[f'Wgate{l}'])
        up = (hn @ p[f'Wup{l}'])
        neuron_act = F.silu(gate) * up
        mlp_out = neuron_act @ p[f'Wdown{l}']
        x = x + mlp_out
    logits = rmsnorm(x[-1:])[0] @ p['Whead']
    return logits


def eval_acc(p, n=60):
    correct = 0
    with torch.no_grad():
        for _ in range(n):
            seq, label = make_example()
            logits = forward(p, seq)
            if logits.argmax().item() == label:
                correct += 1
    return correct / n


def run_global():
    p = init_block()
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(list(p.values()), lr=LR_GLOBAL)
    acc_hist = []
    for step in range(STEPS):
        seq, label = make_example()
        logits = forward(p, seq)
        loss = F.cross_entropy(logits[None], torch.tensor([label]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0); opt.step()
        if step % 20 == 0:
            acc_hist.append(eval_acc(p))
    return acc_hist


def run_local():
    p = init_block()
    frozen_keys = [f'Wo{l}' for l in range(LAYERS)] + [f'Wdown{l}' for l in range(LAYERS)]
    fixed = {k: p[k].clone() for k in frozen_keys}
    local_params = {k: v.clone().requires_grad_(True) for k, v in p.items() if k not in frozen_keys}
    opt = torch.optim.Adam(list(local_params.values()), lr=LR_LOCAL)
    acc_hist = []
    for step in range(STEPS):
        seq, label = make_example()
        pp = dict(local_params); pp.update(fixed)
        logits = forward(pp, seq)
        loss = F.cross_entropy(logits[None], torch.tensor([label]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0); opt.step()
        if step % 20 == 0:
            pp_eval = dict(local_params); pp_eval.update(fixed)
            acc_hist.append(eval_acc(pp_eval))
    return acc_hist


if __name__ == "__main__":
    print("=== GLOBAL backprop (ceiling) ===", flush=True)
    g = run_global()
    print(f"accuracy over training: {[f'{a:.2f}' for a in g]}", flush=True)
    print(f"final: {g[-1]:.2%}", flush=True)

    print("\n=== LOCAL (Wo/Wdown frozen random, per-head/per-neuron-group delta rule) ===", flush=True)
    l = run_local()
    print(f"accuracy over training: {[f'{a:.2f}' for a in l]}", flush=True)
    print(f"final: {l[-1]:.2%}", flush=True)

    print(f"\n=== VERDICT === global={g[-1]:.2%}  local={l[-1]:.2%}  gap={g[-1]-l[-1]:+.2%}", flush=True)
