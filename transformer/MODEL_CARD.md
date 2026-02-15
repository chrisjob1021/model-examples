# Transformer Model Card

## Overview

GPT-2 style decoder-only transformer, incrementally extended toward GPT-OSS architecture.
Two phases: train GPT-2 from scratch, then add modern components one at a time at 7B scale.

## Phase 1: GPT-2 (124M) — Full Training Run

Implement GPT-2 from scratch, validate by loading `openai-community/gpt2` weights, then train.

| Parameter            | Value            |
|----------------------|------------------|
| Layers               | 12               |
| Hidden size          | 768              |
| Attention heads      | 12               |
| Head dimension       | 64               |
| FFN inner dim        | 3072 (4x)        |
| Context length       | 1024             |
| Vocab size           | 50,257           |
| Activation           | GELU             |
| Normalization        | Pre-LayerNorm    |
| Position encoding    | Learned absolute |
| Bias                 | Yes (all layers) |
| Parameters           | 124M             |

### Phase 1 Training Stages

#### Stage 1: Pretraining
General web-scale language modeling.

| Setting       | Value                                                                            |
|---------------|----------------------------------------------------------------------------------|
| Dataset       | [allenai/dolma](https://hf.co/datasets/allenai/dolma) (3T+ tokens, ODC-BY)      |
| Subset        | Sample ~10-20B tokens (feasible for 124M on consumer GPU)                        |
| Objective     | Autoregressive next-token prediction                                             |
| Context       | 1024 tokens                                                                      |

#### Stage 2: Mid-training (Annealing)
High-quality data mix to sharpen capabilities. Learning rate decays toward zero.

| Setting       | Value                                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------------|
| Dataset       | [allenai/dolma3_dolmino_mix-10B-1025](https://hf.co/datasets/allenai/dolma3_dolmino_mix-10B-1025) (ODC-BY)         |
| Content       | Curated subset: math, code, science, high-quality web (same mix used for OLMo 3 stage 2 micro-anneals)            |
| Schedule      | Cosine decay to 0, short relative to pretraining                                                                   |

#### Stage 3: Post-training (SFT)
Instruction following and chat capability.

| Setting       | Value                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| Dataset       | [allenai/tulu-3-sft-mixture](https://hf.co/datasets/allenai/tulu-3-sft-mixture) (ODC-BY)            |
| Content       | FLAN, OpenAssistant, math (MetaMathQA, personas), code, instruction following, safety (WildGuardMix) |
| Format        | Multi-turn chat with system/user/assistant roles                                                     |

---

## Phase 2: Architecture Progression (7B scale, 8x H100)

Validate GPT-OSS-20B architecture by loading `openai/gpt-oss-20b` weights, then train
each architectural change from GPT-2 baseline at 7B scale.

### GPT-OSS-20B Reference Architecture

| Parameter               | GPT-OSS-20B Value           |
|-------------------------|-----------------------------|
| Layers                  | 24                          |
| Hidden size             | 2880                        |
| Attention heads         | 64                          |
| KV heads (GQA)          | 8                           |
| Head dimension          | 64                          |
| Experts                 | 32, top-4                   |
| Expert FFN dim          | 2880 (SwiGLU)               |
| Context length          | 131K (YaRN from 4096 base)  |
| Vocab size              | 201,088                     |
| RoPE theta              | 150,000                     |
| Attention pattern       | Alternating sliding(128)/full |
| Activation              | SiLU (SwiGLU, limit=7.0)    |
| Normalization           | RMSNorm                     |
| Quantization            | MXFP4 (expert weights only) |
| Total params            | 20.9B                       |
| License                 | Apache 2.0                  |

### Training Strategy: Continue from Checkpoint

Each step modifies the architecture and continues training from the previous checkpoint.
This requires **checkpoint surgery** — converting weights from the old shape to the new one.

| Step | Weight Surgery                                        | Expected Loss Spike |
|------|-------------------------------------------------------|---------------------|
| 1    | Copy LayerNorm γ → RMSNorm γ, drop all biases        | Minimal             |
| 2    | Reinitialize FFN (shape mismatch: 2 → 3 matrices)    | Moderate            |
| 3    | Drop position embedding table, no new params          | Moderate            |
| 4    | Mean-pool groups of 4 KV heads → 1 KV head           | Small               |
| 5    | No weight changes — only attention mask changes       | None                |
| 6    | Copy dense FFN → all 32 experts, random init router   | Moderate            |

After each surgery, continue training with a **warmup restart** (re-warmup learning rate
over ~1% of that step's token budget) to let the model adapt to the new component.

### Data & Schedule

Phase 2 is one continuous pretraining run on Dolma, interrupted for surgery at intervals.
Mid-training and post-training happen once, after Step 6.

All 6 steps (including the baseline) are **pretraining only**. Mid-training and SFT
happen once after Step 6 — doing them earlier would be wasted work since surgery
disrupts fine-tuned behaviors.

| | Pretraining | Mid-training | SFT / RL |
|---|---|---|---|
| **Phase 1** (GPT-2 124M) | Yes | Yes | Yes |
| **Phase 2** baseline + Steps 1-6 | Yes (continuous) | After Step 6 | After Step 6 |

**Dataset:** [allenai/dolma3_mix-6T](https://hf.co/datasets/allenai/dolma3_mix-6T) (ODC-BY).
Each step trains on a **fresh, non-overlapping slice** — one continuous pass through
the corpus. No data is repeated between steps.

**Token allocation (7 days total, 8x H100, ~200B tokens at 7B dense):**

| Step                    | Tokens | Wall Time | Rationale                              |
|-------------------------|--------|-----------|----------------------------------------|
| Baseline (GPT-2-7B)    | ~80B   | ~2.5 days | Establish foundation from scratch      |
| Step 1 (RMSNorm)        | ~5B    | ~4 hours  | Near-zero recovery needed              |
| Step 2 (SwiGLU)         | ~25B   | ~20 hours | FFN reinitialized, needs to relearn    |
| Step 3 (RoPE)           | ~20B   | ~16 hours | Attention relearns position encoding   |
| Step 4 (GQA)            | ~10B   | ~8 hours  | Mean-pooled KV, fast recovery          |
| Step 5 (Sliding window) | ~5B    | ~4 hours  | No weight change, just adaptation      |
| Step 6 (MoE)            | ~55B   | ~1.8 days | Router specialization + expert divergence |
| **Total**               |**~200B**| **~7 days**|                                      |

Note: Step 6 (MoE) changes active param count from 7B to ~2-3B, so tokens/second
increases — the 55B token budget accounts for this speedup.

**Learning rate schedule:** Continuous cosine decay with warmup restarts at each surgery.

```
LR
 ^
 |  /\
 | /  \___/\___     /\__________
 |/        \   \___/             \___→ 0
 +----------------------------------------→ tokens
   baseline  s1 s2  s3  s4 s5    s6
```

- **Baseline:** Standard warmup (2K steps) → cosine decay.
- **Steps 1-5:** At each surgery point, re-warmup over ~1% of that step's tokens,
  then resume cosine decay.
- **Step 6:** Final long cosine decay to near-zero LR. This is the last and longest
  step, so the model finishes strong.

**After Step 6 completes:**

| Stage        | Dataset                              | Size     | Purpose                     |
|--------------|--------------------------------------|----------|-----------------------------|
| Mid-training | allenai/dolma3_dolmino_mix-100B-1025 | 50-100B  | High-quality anneal         |
| SFT          | allenai/tulu-3-sft-mixture           | ~326K ex | Instruction following       |
| DPO          | allenai/olmo-2-1124-7b-preference-mix| ~200K ex | Preference alignment        |
| GRPO         | GSM8K + MATH + HumanEval + MBPP      | ~30K prompts | CoT reasoning         |

### Validation Protocol (per step)

After each checkpoint surgery, before committing to a full training run:

1. **Forward pass sanity** (seconds): Run 1 batch through the surgered model.
   Loss should be elevated but finite. If `inf`/`nan`, the weight mapping is broken.

2. **Recovery run** (~1-2K to ~10K steps depending on step): Train briefly and
   confirm loss trends downward toward pre-surgery levels. If loss plateaus far
   above pre-surgery, the surgery introduced a destructive discontinuity.

3. **Parity check** (after full continued training): Compare eval perplexity against
   the previous step on a held-out subset. Each step should be neutral or better.

| Step | Recovery Steps | Key Diagnostic                                |
|------|---------------|-----------------------------------------------|
| 1    | ~100          | Loss nearly unchanged after surgery           |
| 2    | ~5-10K        | Loss spikes then beats pre-surgery (SwiGLU wins) |
| 3    | ~2-5K         | Test extrapolation beyond training context    |
| 4    | ~1-2K         | Compare inference KV cache size               |
| 5    | ~0            | Test long-sequence perplexity + memory usage  |
| 6    | ~5-10K        | Expert utilization + router entropy           |

### Architecture Change Order

Each step adds one change to the previous step's architecture, continuing from its checkpoint.
Order is from simplest/most independent to most complex/interdependent.

#### Baseline: GPT-2 scaled to 7B

| Parameter       | Value      |
|-----------------|------------|
| Layers          | 32         |
| Hidden size     | 4096       |
| Attention heads | 32         |
| Head dimension  | 128        |
| FFN inner dim   | 16,384 (4x)|
| Vocab size      | 50,257     |
| Activation      | GELU       |
| Normalization   | Pre-LayerNorm |
| Position        | Learned absolute |
| Bias            | Yes        |
| **Total params**| **~6.8B**  |

---

#### Step 1: RMSNorm + Drop Biases

**What changes:** LayerNorm → RMSNorm, remove all bias terms.

**Why first:** Pure drop-in swap with zero interaction with other components.
RMSNorm skips the mean subtraction in LayerNorm — only normalizes by variance.
Fewer parameters, faster computation, same or better training stability at scale.

**Why remove biases:** Modern architectures (Llama, GPT-OSS, PaLM) all dropped them.
At scale, biases add parameters without measurable quality gain.

| Changed         | Before          | After           |
|-----------------|-----------------|-----------------|
| Normalization   | Pre-LayerNorm   | Pre-RMSNorm     |
| Bias (attn/FFN) | Yes             | No              |
| Norm params     | 2 * 2 * d       | 2 * d per layer |

**Checkpoint surgery:**
- Copy LayerNorm `weight` (γ) → RMSNorm `weight` (γ). Both are shape `[d_model]`.
- LayerNorm `bias` (β) is dropped — RMSNorm has no bias.
- All attention and FFN bias vectors are dropped.
- All other weights (attention projections, FFN matrices, embeddings) transfer 1:1.

**Validation:**
1. Forward pass: loss should be close to pre-surgery baseline (not `inf`/`nan`).
2. Recovery run (~100 steps): loss trends downward immediately.
3. **Expected recovery: ~100 steps.** Dropping biases and β is a small perturbation.

---

#### Step 2: SwiGLU Activation

**What changes:** GELU FFN → SwiGLU FFN.

**Why second:** Only touches the FFN block. Independent of attention and position encoding.

GPT-2 FFN: `y = W₂ · GELU(W₁ · x)`  (2 matrices)
SwiGLU FFN: `y = W_down · (SiLU(W_gate · x) ⊙ W_up · x)`  (3 matrices)

The gating mechanism (element-wise multiply of two projections) gives better gradient
flow. 3 matrices instead of 2, so reduce intermediate dim to maintain param count.

| Changed          | Before             | After                    |
|------------------|--------------------|--------------------------|
| Activation       | GELU               | SiLU (SwiGLU gate)       |
| FFN matrices     | 2 (up, down)       | 3 (gate, up, down)       |
| FFN intermediate | 16,384 (4x hidden) | 11,008 (~2.7x, keeps 7B) |

**Checkpoint surgery:**
- Old FFN: `W1 [4096, 16384]`, `W2 [16384, 4096]` — 2 matrices.
- New SwiGLU: `W_gate [4096, 11008]`, `W_up [4096, 11008]`, `W_down [11008, 4096]` — 3 matrices.
- Shapes don't match (16384 → 11008, 2 → 3 matrices). **No clean weight mapping.**
- Strategy: Reinitialize all 3 FFN matrices. Attention weights carry over intact.
  Alternatively, slice W1 into W_gate and W_up (first 11008 dims each) and slice W2
  into W_down — preserves some knowledge but is lossy.

**Validation:**
1. Forward pass: loss will spike significantly (FFN is ~2/3 of each layer's params).
2. Recovery run (~5-10K steps): loss should trend toward and eventually beat pre-surgery
   levels, since SwiGLU is a strictly better FFN architecture.
3. **Expected recovery: ~5-10K steps.** This is the roughest surgery — attention carries
   the model while FFN relearns.

---

#### Step 3: RoPE (Rotary Position Embeddings)

**What changes:** Remove learned positional embedding table, apply rotary embeddings
to Q and K vectors in attention.

**Why third:** Changes how attention computes position — must come before sliding
window or YaRN context extension. Independent of FFN changes (steps 1-2).

RoPE encodes position by rotating Q/K vectors in 2D subspaces. Relative position
information emerges naturally from the dot product of rotated vectors.

| Changed              | Before                    | After               |
|----------------------|---------------------------|----------------------|
| Position encoding    | Learned absolute (table)  | RoPE (computed)      |
| Embedding params     | vocab + position tables   | vocab table only     |
| RoPE theta           | —                         | 150,000 (GPT-OSS)   |
| Extrapolation        | None (fixed 1024)         | Natural (unbounded)  |

**Checkpoint surgery:**
- Delete the learned position embedding table `wpe [1024, 4096]`.
- RoPE has no learned parameters — it's computed on the fly from position indices.
- All other weights (token embeddings, attention, FFN) transfer 1:1.
- The attention Q/K weights were trained to expect absolute position info summed into
  the residual stream. Now they must learn to use rotary-encoded relative positions.

**Validation:**
1. Forward pass: loss spikes (attention has no position signal until it adapts).
2. Recovery run (~2-5K steps): attention weights adapt to rotary encoding.
3. After recovery, test extrapolation: evaluate on sequences longer than training
   context. Loss should degrade gracefully rather than collapse.
4. **Expected recovery: ~2-5K steps.** At this point, dense components match Llama.

---

#### Step 4: Grouped-Query Attention (GQA)

**What changes:** Reduce the number of KV heads. Multiple query heads share KV heads.

**Why fourth:** Modifies the attention head structure. Must be stable before
adding sliding window patterns on top.

MHA: 32 Q heads, 32 KV heads (1:1 ratio)
GQA: 32 Q heads, 8 KV heads (4:1 ratio) — each KV head serves 4 query heads

| Changed          | Before            | After                |
|------------------|-------------------|----------------------|
| Q heads          | 32                | 32                   |
| KV heads         | 32                | 8                    |
| KV params/layer  | 2 × 4096 × 4096  | 2 × 4096 × 1024     |
| KV cache (infer) | 32 × d_head       | 8 × d_head (4x less)|

**Freed params:** ~800M total from KV reduction. Redistribute to more layers (36)
or larger FFN to maintain 7B.

**Checkpoint surgery:**
- For each layer, K and V projections go from `[4096, 4096]` → `[4096, 1024]`.
- Group every 4 adjacent KV heads and **mean-pool** their weights into 1 KV head.
  E.g., new K head 0 = mean(old K heads 0-3). This preserves the most information.
- Q and O projections transfer 1:1 (still 32 heads).
- If redistributing freed params to more layers: new layers are randomly initialized.

**Validation:**
1. Forward pass: loss slightly elevated (mean-pooled KV is a good but lossy init).
2. Recovery run (~1-2K steps): fast recovery since mean-pooling is a strong init.
3. Compare inference speed: KV cache is 4x smaller, should see measurable speedup.
4. **Expected recovery: ~1-2K steps.** Mean-pooling is well-studied (Llama 2 paper).

---

#### Step 5: Alternating Sliding Window + Full Attention

**What changes:** Half the layers use sliding window attention (local, window=128),
alternating with full causal attention layers.

**Why fifth:** Builds on the finalized attention mechanism (GQA + RoPE). This is
a GPT-OSS-specific pattern — not present in Llama.

Sliding window layers only attend to the nearest 128 tokens — O(n × w) instead of O(n²).
Full attention layers still see the entire sequence, acting as information aggregators.
The alternating pattern gives both local precision and global coherence.

| Changed              | Before                   | After                         |
|----------------------|--------------------------|-------------------------------|
| Attention pattern    | Full causal (all layers) | Alternating: sliding(128)/full|
| Memory (long seqs)   | O(n²) all layers         | O(n×128) half + O(n²) half    |
| Layer types          | Uniform                  | [sliding, full, sliding, ...]  |

**Checkpoint surgery:**
- No weight changes whatsoever. Only the attention mask changes.
- Odd-numbered layers get a sliding window mask (attend to nearest 128 tokens).
- Even-numbered layers keep full causal attention.

**Validation:**
1. Forward pass: loss should be near-identical to pre-surgery (full causal is a
   superset of sliding window — restricting it barely hurts on short sequences).
2. No recovery run needed — **0 steps to recover.**
3. Test on long sequences: perplexity should hold while attention memory drops ~50%.

---

#### Step 6: Mixture of Experts (MoE)

**What changes:** Replace each dense SwiGLU FFN with a routed set of expert FFNs.
Only top-k experts are activated per token.

**Why last:** Most complex structural change. Affects parameter accounting, training
dynamics, parallelism strategy, and inference. Everything else should be stable first.

Each token goes through a learned router that selects the top-k experts.
The rest of the experts are skipped, making compute proportional to active params.

| Changed               | Before (dense)    | After (MoE)                   |
|-----------------------|-------------------|-------------------------------|
| FFN per layer         | 1 dense           | 32 experts, top-4 active      |
| FFN intermediate      | 11,008            | 2,880 per expert (GPT-OSS)    |
| Router                | None              | Linear(d_model, num_experts)  |
| Aux loss              | None              | Load balancing (coeff=0.01)   |
| Active params         | 7B                | ~2-3B active / 7B+ total      |
| Training parallelism  | FSDP              | FSDP + expert parallelism     |

**Note:** This step changes the 7B parameter accounting. To keep 7B total, each
expert is smaller than the dense FFN it replaces. To keep 7B *active*, the total
model grows to ~20B+. Choose based on your comparison goal.

**Checkpoint surgery:**
- Each of the 32 expert FFNs is initialized as a **copy of the dense SwiGLU FFN**.
  If expert intermediate dim differs from dense (2,880 vs 11,008), slice the weights.
- Router `Linear(4096, 32)` is **randomly initialized** — no prior knowledge of
  which tokens should go where.
- All attention weights, norms, and embeddings transfer 1:1.
- All experts start identical. Specialization emerges during training as the router
  learns to route different token types to different experts.

**Validation:**
1. Forward pass: loss should be close to pre-surgery (all experts are copies of the
   working dense FFN, so any top-4 selection produces the same output initially).
2. Recovery run (~5-10K steps): router learns non-trivial routing, experts diverge.
3. Monitor **expert utilization**: track how many tokens each expert receives per batch.
   If any expert gets <1% of tokens, it's dying — increase aux loss coefficient.
4. Monitor **router entropy**: should start high (random) and settle to moderate
   (specialized but not collapsed).
5. **Expected recovery: ~5-10K steps.** Initial loss is fine (expert clones), but
   the model needs time to develop meaningful specialization.

---

### Post-Architecture Steps

After all 6 steps, the architecture matches GPT-OSS-20B (scaled to 7B).

#### YaRN Context Extension
- Apply YaRN RoPE scaling (factor=32, from 4096 → 131K)
- Continue training on long documents
- Dataset: [allenai/dolma3_longmino_mix-50B-1025](https://hf.co/datasets/allenai/dolma3_longmino_mix-50B-1025)

#### MXFP4 Quantization
- Quantize expert FFN weights to MXFP4 (4.25 bits/param)
- Keep attention, router, and embeddings in BF16
- Quantization-aware training or post-training quantization

#### Post-training

Three stages, run sequentially. Each builds on the previous checkpoint.

**Stage 1: SFT (Supervised Fine-Tuning)**

Teach the model to follow instructions and respond in chat format.

| Setting  | Value                                                                              |
|----------|-------------------------------------------------------------------------------------|
| Dataset  | [allenai/tulu-3-sft-mixture](https://hf.co/datasets/allenai/tulu-3-sft-mixture)   |
| Content  | FLAN, OpenAssistant, math, code, instruction following, safety                     |
| Format   | Multi-turn chat (system/user/assistant)                                            |
| Special  | Add `<think>`, `</think>` tokens to vocabulary and tokenizer (resize embeddings)   |

**Stage 2: DPO (Preference Alignment)**

Align the model to prefer helpful, harmless, and high-quality responses over poor ones.
DPO is simpler than PPO — no reward model needed, just preference pairs.

| Setting  | Value                                                                                                    |
|----------|----------------------------------------------------------------------------------------------------------|
| Dataset  | [allenai/olmo-2-1124-7b-preference-mix](https://hf.co/datasets/allenai/olmo-2-1124-7b-preference-mix)   |
| Content  | Chosen/rejected response pairs across general, math, code, safety                                        |
| Method   | Direct Preference Optimization (DPO)                                                                     |
| β        | 0.1 (standard, controls deviation from SFT policy)                                                       |

**Stage 3: GRPO (Reasoning via Verifiable Rewards)**

Teach the model to reason using chain-of-thought. This is the DeepSeek-R1 approach —
no reward model needed, just problems with verifiable answers.

| Setting          | Value                                                                    |
|------------------|--------------------------------------------------------------------------|
| Method           | Group Relative Policy Optimization (GRPO)                                |
| Math prompts     | GSM8K (~8K), MATH (~12K), NuminaMath                                     |
| Code prompts     | HumanEval, MBPP, LiveCodeBench                                          |
| Reward signal    | Outcome-based: did the final answer match? did the code pass tests?      |
| Generations/prompt | 8-16 completions per prompt, ranked by correctness                     |
| Think tokens     | Model learns when and how long to use `<think>...</think>` traces        |

How GRPO works:
1. For each prompt, generate k completions (e.g., 16).
2. Score each completion with a verifiable reward (correct answer = 1, wrong = 0).
3. Compute advantage relative to the group mean (no separate reward model).
4. Update policy to increase probability of high-reward completions.

The model discovers chain-of-thought on its own — rewarding correct answers
incentivizes the model to "think before answering" using the `<think>` tokens.

**Why this order (SFT → DPO → GRPO):**
- SFT first: the model must follow instructions before it can be aligned or reason.
- DPO second: establishes general quality preferences and safety guardrails.
- GRPO last: reasoning is the highest-level capability, built on top of instruction
  following and alignment. Running GRPO before DPO risks the model learning to
  reason well but producing unsafe or unhelpful outputs.

**Reference:** GPT-OSS used "similar CoT RL techniques as o3." DeepSeek-R1 demonstrated
that GRPO alone (without a reward model) is sufficient to elicit strong reasoning.

---

## Compute Budget

### Phase 1: GPT-2 (124M)

Trainable on a single GPU (L40S, A100, or even consumer 3090/4090).

| Stage         | Tokens  | Time estimate (1x L40S) |
|---------------|---------|-------------------------|
| Pretraining   | 10-20B  | 1-3 days                |
| Mid-training  | 1-2B    | ~4 hours                |
| Post-training | ~100M   | ~30 min                 |

### Phase 2: Architecture Steps (7B, 8x H100)

| Metric                    | Value              |
|---------------------------|---------------------|
| GPUs                      | 8x H100 SXM        |
| GPU memory                | 80GB each           |
| Effective TFLOPS (~45%)   | ~3,500              |
| Per step (7-day run)      | ~200B tokens (dense)|
| Per step (1-day ablation) | ~30B tokens         |
| Parallelism               | FSDP + expert parallel (MoE step) |

### Inference (1x L40S, 48GB)

| Model           | Format  | VRAM    | Fits? |
|-----------------|---------|---------|-------|
| GPT-2 (124M)    | FP32    | ~0.5 GB | Yes   |
| 7B dense        | BF16    | ~14 GB  | Yes   |
| 7B dense        | INT4    | ~3.5 GB | Yes   |
| GPT-OSS-20B ref | MXFP4   | ~14 GB  | Yes   |
| GPT-OSS-20B ref | FP16    | ~42 GB  | Tight |

---

## Datasets (Allen AI)

| Stage            | Dataset                              | Size    | License | Purpose                        |
|------------------|--------------------------------------|---------|---------|--------------------------------|
| Pretraining      | allenai/dolma                        | 3T+ tok | ODC-BY  | Web-scale language modeling    |
| Pretraining (v3) | allenai/dolma3_mix-6T                | 6T tok  | ODC-BY  | Latest pretraining mix (OLMo 3)|
| Mid-training     | allenai/dolma3_dolmino_mix-10B-1025  | 10B tok | ODC-BY  | High-quality anneal (micro)    |
| Mid-training     | allenai/dolma3_dolmino_mix-100B-1025 | 100B tok| ODC-BY  | High-quality anneal (full)     |
| Long context     | allenai/dolma3_longmino_mix-50B-1025 | 50B tok | ODC-BY  | Long document training         |
| Post-train (SFT) | allenai/tulu-3-sft-mixture           | ~326K   | ODC-BY  | Instruction tuning             |
| Post-train (DPO) | allenai/olmo-2-1124-7b-preference-mix| ~200K   | ODC-BY  | Preference alignment (7B)      |
| Post-train (GRPO)| GSM8K + MATH + NuminaMath            | ~30K    | MIT/varies | Math reasoning prompts      |
| Post-train (GRPO)| HumanEval + MBPP + LiveCodeBench     | ~2K     | MIT/varies | Code reasoning prompts      |

---

## Reference Models

| Model                  | Params  | Active | Purpose                                     | License    |
|------------------------|---------|--------|---------------------------------------------|------------|
| openai-community/gpt2  | 124M    | 124M   | Phase 1: implement & validate basic transformer | MIT        |
| openai/gpt-oss-20b     | 20.9B   | ~4B    | Phase 2: validate GPT-OSS architecture      | Apache 2.0 |
| allenai/OLMoE-1B-7B    | 6.9B    | 1.3B   | MoE routing reference (64 experts, top-8)   | Apache 2.0 |
