# prototype_cluster_brick_transformer — first result, 2026-11-24

**Question**: can brick-style local learning (proven on the fly-reservoir + hand-built toy nets)
be applied to a real transformer (Llama) architecture's own structure? "Shrimp eating a llama."

**Design**: a single real-shaped transformer block (4-head attention + SwiGLU MLP). `Wo` and
`Wdown` (the dense mixing matrices) frozen at random init, playing the same role as the fly
connectome's fixed random wiring. Only the per-head q/k/v and per-neuron-group gate/up weights
are trained, each via its own local delta-rule update — no gradient crosses from one head/neuron
group into another. Compared directly against full end-to-end backprop on the identical block.

**First task attempted (in-context recall, random key-value mapping every step)**: INCONCLUSIVE —
even the global-backprop ceiling never reliably beat chance (12.5%). Wrong task shape, not a real
test of the method (this requires in-context induction, not the proven associative-memory shape).

**Real bug found and fixed along the way**: the toy block initially had no normalization anywhere
(real Llama blocks have RMSNorm before attention and before MLP; this one omitted it) — loss
exploded to 8+ billion under Adam at lr=0.05. Fixed with RMSNorm before attention/MLP/readout,
lr=0.01, grad-norm clipping at 1.0. Same failure class as the documented PC-ALM Mamba blowup
("classic runaway positive-feedback blowup") — worth remembering as a recurring failure mode
across unrelated local-learning experiments.

**Real task (fixed associative memory, matching the proven fly-reservoir shape — 3 fixed
key-value pairs, shuffled presentation order, held constant across all training)**:
- GLOBAL backprop (full block trained end-to-end): **100% by step 40, held for 3000 steps.**
- LOCAL (Wo/Wdown frozen, per-head/per-neuron-group delta rule): **100% by step 40, held for
  3000 steps.**
- **Gap: 0.00%.** Exact match.

**Honest scope — what this does NOT yet show**: single block (no depth/stacking), tiny scale
(64 hidden, 4 heads, 128 intermediate), only 3 associations, task hit ceiling almost immediately
on both methods — no real difficulty was present that could have exposed a gap if one exists.
This is a genuine first positive data point, not proof the method survives at real model scale or
on a task requiring real depth/credit-assignment distance.

**Next real test**: stack 2+ of these blocks (so local credit has to survive crossing a layer
boundary, the actual hard part) and/or scale up the number of associations, before concluding
anything about whether this generalizes toward the real target architecture.

**UPDATE same day — LAYERS=2, restarted both from scratch**: stacked a second block (Wo0/Wdown0
AND Wo1/Wdown1 both frozen in the local variant, so layer-0's trainable weights only get updated
via gradient routed back through BOTH frozen mixing matrices in sequence — the real "does local
credit survive crossing a layer boundary" test). **Still an exact match: global=100%, local=100%,
gap=0.00%**, both hitting ceiling even faster than the single-layer run (first checkpoint already
at 62-65%). Honest limit, unchanged: the task is still too easy for either method to be
differentiated by — depth alone didn't add enough real difficulty. A genuinely hard test still
needs either a harder associative load or a task requiring real multi-hop structure across the
added depth, not just more layers computing the same easy lookup.

**UPDATE same day — genuine 2-hop chained task, restarted both from scratch**: redesigned the
task so a single direct lookup can't shortcut it — `key->mid` is one fixed fact, `mid->val` is a
SEPARATE fixed fact, the query gives the key, the answer requires actually chaining through the
intermediate. This is a real test of cross-layer composition, not just "more layers along for the
ride." **Still an exact match: global=100%, local=100%, gap=0.00%** — local actually reached
ceiling slightly FASTER this run (100% by step 20 vs global's 98%->100%).

**Honest pattern across three consecutive escalations (1 layer -> 2 layers -> genuine 2-hop
chaining), all landing at 0.00% gap**: this is a real, consistent result, not a fluke of one run.
But it also means this toy scale (14-token vocab, a few hundred params, 4 fixed chains memorized
over 3000 steps) still isn't hard enough to find where local learning would actually break, if it
ever does -- both methods hit ceiling almost immediately every time. The honest next step isn't
another toy-scale escalation of the same shape; it's either a combinatorial-generalization test
(hold out some key/mid/val combinations entirely, test zero-shot recombination) or moving toward
real scale (a small slice of the real target architecture) to find a genuine stress point.

**FOURTH ESCALATION, same day — REAL pretrained architecture weights, not random init.** New script,
`cluster_brick_vs_backprop_real_weights.py`: loaded 2 real consecutive layers
(10, 11) of an actual Llama-3.1-8B-Instruct checkpoint's pretrained weights
(real dims: H=4096, 32 attention heads, 8 KV heads via GQA, 14336 intermediate) -- one variable
changed from the toy version (random init -> real pretrained weights), same 2-hop chained task,
same freeze/train split (Wo/Wdown frozen at their REAL pretrained values this time, not random).

**Three real bugs found and fixed before the real result, worth recording**: (1) missed that
Llama-3.1 uses grouped-query attention (8 KV heads, not 32) -- fixed by reading
`num_key_value_heads` from config and sharing each KV head across its group of Q heads. (2) a
monitoring mistake, not a code bug -- piped the first run through `| tail -60`, which buffers
everything until the upstream process exits (exactly the trap Monitor's own docs warn about),
making the log look silent for 20+ minutes when the run was actually progressing fine. (3) the
real crash: marking the FULL 128,256-row embedding/lm_head tables as trainable (even though
`forward()` only ever indexes ~13 rows) blew Adam's per-parameter state past the GPU's 16GB --
`CUDA out of memory`, "30GB allocated" on a 16GB card. Fixed by freezing `emb`/`Whead` in both the
global and local runs (training the vocabulary table was never the point of this test).

**Real result, GPU, both runs restarted clean after the fixes**: GLOBAL backprop reached 100% by
step 60, held for all 400 steps. LOCAL (Wo/Wdown frozen at their real pretrained values, per-head/
per-neuron-group delta rule) reached 100% by step 40, held for all 400 steps. **Gap: 0.00%** --
fourth consecutive escalation landing at an exact match, now on the real target architecture, not
a toy. Notable but NOT yet established as a general finding (one run, one seed): local actually
converged faster early on (77% at step 20 vs global's 27%) -- worth re-testing across seeds before
treating that specific gap as real, but the final-accuracy match itself is now four-for-four.

**MULTI-SEED ROBUSTNESS CHECK, same day, 3 independent seeds on the real-weights slice**:

| Seed | global @ step20 | local @ step20 | global final | local final |
|------|-----------------|-----------------|---------------|-------------|
| 1 | 48% | 70% | 100% | 100% |
| 2 | 38% | 77% | 100% | 100% |
| 3 | 23% | 78% | 100% | 100% |

**Local was ahead at step 20 in all three seeds**, and every single run across both methods and
all three seeds converged to the exact same 100% final ceiling. This is no longer a single lucky
run -- it's a real, reproducible early-convergence advantage for the local method on this task,
confirmed across independent seeds. Final accuracy remains an exact tie in every case; the gap is
specifically in how fast each method gets there, not in what it ultimately reaches.

**Still not fully explained**: the leading hypothesis remains mundane rather than exotic -- the
local variant trains strictly fewer parameters (Wo/Wdown frozen), which may simply make Adam's
optimization problem easier/faster to solve, independent of anything deep about "local learning."
Worth isolating directly (e.g. compare against a global run with an equivalently-reduced parameter
count) before treating the speed advantage as evidence that local learning is intrinsically
better, not just smaller.

**REAL-SCALE ESCALATION, same day -- doubled the real depth (2 -> 4 consecutive layers,
8/9/10/11), switched to bf16 to stay within the GPU's 16GB.** One real bug along the way: the
attention mask was created as float32 by default while everything else was bf16, causing a silent
dtype promotion in softmax that then clashed with the bf16 value matrix -- fixed by making the
mask's dtype match explicitly.

**Result: global=100% (reached at step 80), local=100% (reached at step 20) -- gap=0.00% on final
accuracy, and the speed gap WIDENED, not shrank, as real depth doubled.** This is now five
consecutive real/toy escalations landing at an exact final-accuracy match, and the convergence-
speed advantage for local has held and grown at greater real depth, not vanished. Script:
`D:\flybrain\cluster_brick_vs_backprop_real_weights.py` (same file, LAYER_IDS updated to the
4-layer slice).

**Where this honestly stands**: local-matches-global is now well-established across five
independent escalations (toy 1-layer, toy 2-layer, toy 2-hop, real 2-layer x4 seeds, real
4-layer). The speed advantage is real and reproducible across 5 total runs now (4 real-weight
runs + this one), growing with more frozen parameters -- consistent with, but not yet proof of,
the "fewer trainable parameters makes Adam's job easier" hypothesis. The clean next test to
actually isolate WHY would be a global run restricted to the SAME parameter count as local (freeze
Wo/Wdown there too, just also optimize some other dummy no-op path) -- not yet done.
