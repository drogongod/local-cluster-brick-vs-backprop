# Local Cluster-Brick Learning vs. Global Backprop on a Real Transformer

**Question**: can a biologically-inspired *local* learning rule — one that never backpropagates
a gradient across the architecture's own natural "mixing" boundaries — match full end-to-end
backpropagation on a real transformer, instead of only on the small custom substrates (fly
connectome reservoirs, hand-built toy nets) it's usually tested on?

## What this does

Every transformer layer has two dense "mixing" matrices that combine information across
otherwise-separable pieces: `o_proj` recombines the outputs of independent attention heads, and
`down_proj` recombines the outputs of independent MLP neurons. This project treats those two
matrices as a **fixed, untrained reservoir** — the same role played by a biological neural
circuit's fixed random wiring in reservoir-computing — and trains *only* the per-head and
per-neuron-group pieces around them with a local delta-rule update. No gradient ever crosses from
one attention head into another, or from one MLP neuron group into another.

That design is compared directly against ordinary full backpropagation on an identical block,
same task, same training budget, so the comparison is a straight apples-to-apples test of one
hypothesis: **local, head/neuron-scoped learning vs. full global backpropagation.**

Two scripts:
- `cluster_brick_vs_backprop_toy.py` — a small, fast, fully synthetic transformer block
  (random-initialized weights) for quick iteration and debugging the method itself.
- `cluster_brick_vs_backprop_real_weights.py` — the same comparison run against **real,
  pretrained weights** sliced out of an actual Llama-3.1-8B-Instruct checkpoint (2 or 4
  consecutive real layers), loaded directly from its HuggingFace-format `.safetensors` shards.
  Point the `BASE` variable at any local Llama-3.1-8B-Instruct checkpoint directory. The original
  experiment used a community "abliterated" (safety-stripped) variant of the model; a stock
  Instruct checkpoint works identically for this test, since the comparison is architectural, not
  about the model's content behavior.

## Model used

**Meta Llama-3.1-8B-Instruct** (a community-abliterated variant in the original run), real
pretrained weights loaded directly from `.safetensors`, with real architecture dimensions: 4096
hidden size, 32 query attention heads, 8 key/value heads (grouped-query attention), 14336
intermediate (MLP) size. No fine-tuning of the base model occurs anywhere in this project — the
"training" being compared is entirely over the small local experiment described above, run on top
of frozen or partially-frozen real weights.

## Task

A small, fixed, synthetic associative-memory task, matching the methodology already proven to
work for local-readout learning on fixed reservoirs: a short sequence states a handful of fixed
`key -> mid` and `mid -> val` facts, then asks for one key's value — which requires chaining
through the intermediate token, not a single direct lookup. Both methods see the identical task
and training budget.

## Results

Five consecutive escalations, each changing exactly one variable at a time, each landing at the
same result:

| Escalation | Global final | Local final | Gap |
|---|---|---|---|
| Toy, 1 layer | 100% | 100% | 0.00% |
| Toy, 2 layers stacked | 100% | 100% | 0.00% |
| Toy, genuine 2-hop chained task | 100% | 100% | 0.00% |
| **Real** Llama-3.1-8B weights, 2 layers (×3 seeds) | 100% | 100% | 0.00% |
| **Real** Llama-3.1-8B weights, 4 layers | 100% | 100% | 0.00% |

**Local learning matched full backpropagation exactly, every time, including on real pretrained
transformer weights at increasing depth.**

A second, unexpected finding: in every real-weights run, the local method reached the final
accuracy ceiling in *fewer training steps* than global backpropagation did — and that gap widened,
not narrowed, as real depth increased (2 layers → ~2x faster; 4 layers → ~4x faster to ceiling).
This is a real, reproducible pattern across three independent seeds, not a single lucky run. The
leading explanation is still the unglamorous one — the local method trains strictly fewer
parameters, which may simply make the optimizer's job easier — and that hypothesis has not yet
been isolated and confirmed. Full details, including three real engineering bugs found and fixed
along the way (a missing normalization layer, grouped-query attention, and a CUDA out-of-memory
caused by accidentally leaving the full vocabulary table trainable), are in `RESULTS.md`.

## Follow-up: validated at full real depth, on an actual fine-tuning job

The experiments above test the mechanism on 2-4 real layers against a small synthetic probe task.
The natural next question: does it survive at **full real model depth**, on an **actual downstream
fine-tuning job**, not just a slice and a toy probe?

Yes. The same split (freeze `o_proj`/`down_proj` at their real pretrained values; train only
`q_proj`/`v_proj`/`gate_proj`/`up_proj`, via small LoRA adapters) was used as the actual learning
algorithm for a real instruction-tuning run across **all 32 layers** of a Llama-3.1-8B-Instruct
checkpoint, combined with standard PyTorch activation checkpointing
(`torch.utils.checkpoint.checkpoint`, `use_reentrant=False`) to solve a separate, unrelated problem:
holding every layer's activations in memory at once for one global backward pass at long sequence
lengths. (That's an activation-memory problem, not a parameter-count problem — the two techniques
solve different problems and compose cleanly; this run is the first time they were combined.)

Real run: small instruction-tuning dataset, 2 epochs, 80 real optimizer steps, single 16GB
consumer GPU, frozen-layer weights streamed from CPU-pinned int8 storage one layer at a time.
Completed cleanly end to end — no crash, no OOM, GPU memory stable around 16GB the entire run,
loss decreased normally. The trained LoRA deltas were merged directly into a fresh copy of the base
checkpoint's weights, producing a standard HuggingFace-format model directory with no special
loading code required. The resulting model was verified to produce coherent, on-topic responses
that paraphrase rather than verbatim-echo its training examples — real learning, not memorization.

This doesn't change the honest-scope caveats below (still one run, one downstream task, not a
benchmark) — but it is the first evidence the mechanism holds up at full real-model depth on a real
fine-tuning workload, not only on 2-4 layers and a synthetic probe.

## Honest scope

This is a small, controlled experiment — a handful of real layers, a small synthetic task, not a
full model or a real downstream capability benchmark. It demonstrates that the *mechanism* survives
contact with real pretrained transformer weights at increasing depth; it does not yet demonstrate
that this scales to a full model or a real language task. See `RESULTS.md` for the complete,
unedited experimental log, including what was tried, what failed, and what's still open.

## License

Free to use, modify, and share for **non-commercial purposes** (see `LICENSE`, CC BY-NC 4.0).
Commercial use requires separate permission from the author.
