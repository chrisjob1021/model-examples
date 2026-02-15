# Transformer (GPT-2 124M)

A from-scratch PyTorch implementation of GPT-2 (124M parameters) with a 3-stage training pipeline: pretraining, mid-training annealing, and supervised fine-tuning.

## Files

| File | Description |
|------|-------------|
| `gpt2.py` | Model architecture — `GPT2Config`, `ManualCausalSelfAttention`, `GPT2MLP`, `GPT2Block`, `GPT2`, and `GPT2Trainer`. Includes both a manual attention path (educational) and a `scaled_dot_product_attention` path (production). |
| `train_gpt2.py` | 3-stage training script. Stage 1: pretraining on [Dolma](https://huggingface.co/datasets/allenai/dolma). Stage 2: mid-training annealing on [dolma3_dolmino_mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025). Stage 3: SFT on [Tulu-3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture). |
| `evaluate_model.py` | Perplexity evaluation on WikiText-2 and optional text generation from prompts. |
| `upload_to_huggingface.py` | Upload a trained checkpoint and auto-generated model card to HuggingFace Hub. |
| `MODEL_CARD.md` | Design document describing the training plan and architecture decisions. |

## Architecture overview

```
Input token IDs  (B, T)
        │
   ┌────▼────┐
   │   wte   │  Token embedding   (50257, 768)
   └────┬────┘
        +  ◄── wpe: Position embedding (1024, 768)
        │
   ┌────▼────┐
   │ Dropout  │
   └────┬────┘
        │
        ▼
  ╔═══════════╗  ×12 blocks
  ║  GPT2Block ║
  ║  ┌───────┐ ║
  ║  │ LN_1  │ ║  Pre-LayerNorm
  ║  │ Attn  │ ║  Multi-head causal self-attention (12 heads, 64 dim each)
  ║  │ + res  │ ║  Residual connection
  ║  │ LN_2  │ ║  Pre-LayerNorm
  ║  │ MLP   │ ║  FFN: 768 → 3072 → 768 with GELU
  ║  │ + res  │ ║  Residual connection
  ║  └───────┘ ║
  ╚═════╤═════╝
        │
   ┌────▼────┐
   │  ln_f   │  Final LayerNorm
   └────┬────┘
   ┌────▼────┐
   │ lm_head │  Linear (768 → 50257), weight-tied with wte
   └────┬────┘
        ▼
   Logits  (B, T, 50257)
```

Key design choices:
- **Pre-LayerNorm** (normalize before each sublayer, not after)
- **Weight tying** between token embedding (`wte`) and output projection (`lm_head`)
- **Conv1D ↔ Linear transpose** handled in `from_huggingface()` for HuggingFace weight compatibility
- **Manual vs. builtin attention** toggled with `use_builtin_attn` (defaults to `True` for fused-kernel speed)

## Unit testing with HuggingFace GPT-2 weights

The implementation can be validated by loading the official `openai-community/gpt2` weights from HuggingFace and comparing outputs against the reference model. This confirms that every layer — embeddings, attention, MLP, LayerNorm, and the output projection — is wired correctly.

### Quick validation (CLI)

The `--validate-weights` flag runs a built-in end-to-end test:

```bash
cd transformer/
python train_gpt2.py --validate-weights
```

This will:
1. Download the HuggingFace `openai-community/gpt2` checkpoint
2. Load weights into both the HuggingFace reference model and our implementation (transposing Conv1D weights as needed)
3. Feed an identical prompt through both models
4. Compare logits element-wise (pass threshold: `atol=1e-4`)
5. Generate a text sample as a qualitative sanity check

Expected output:

```
Logit comparison (prompt: 'The meaning of life is'):
  Max absolute difference:  X.XXe-07
  Mean absolute difference: X.XXe-07
  PASS: logits match within atol=0.0001

Generated: The meaning of life is ...

Weight validation successful!
```

### Writing your own tests

You can also test individual components or run the comparison programmatically:

```python
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from gpt2 import GPT2

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Load the same weights into both implementations
hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
our_model = GPT2.from_huggingface("openai-community/gpt2", device=device).eval()

# Encode a test prompt
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt").to(device)

with torch.no_grad():
    hf_logits = hf_model(input_ids).logits
    our_logits = our_model(input_ids)

# Verify outputs match
assert torch.allclose(hf_logits, our_logits, atol=1e-4), (
    f"Max diff: {(hf_logits - our_logits).abs().max().item():.2e}"
)
print("PASS")
```

#### Testing the manual attention path

By default the model uses PyTorch's fused `scaled_dot_product_attention`. To verify the hand-written attention loop produces the same results:

```python
from gpt2 import GPT2

builtin_model = GPT2.from_huggingface("openai-community/gpt2", device=device).eval()
manual_model  = GPT2.from_huggingface("openai-community/gpt2", device=device)

# Switch every block to the manual path
for block in manual_model.h:
    block.attn.use_builtin = False
manual_model.eval()

with torch.no_grad():
    logits_builtin = builtin_model(input_ids)
    logits_manual  = manual_model(input_ids)

assert torch.allclose(logits_builtin, logits_manual, atol=1e-5)
print("Manual attention matches builtin")
```

#### Testing text generation

```python
output_ids = our_model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
# Should produce coherent English — a qualitative check that the
# autoregressive loop, causal masking, and softmax sampling all work.
```

### What `from_huggingface()` does

The HuggingFace GPT-2 checkpoint uses `Conv1D` layers that store weights as `(in_features, out_features)` — the transpose of `nn.Linear`'s `(out_features, in_features)`. The `GPT2.from_huggingface()` class method handles this:

1. Downloads the HuggingFace model via `GPT2LMHeadModel.from_pretrained()`
2. Reads its `state_dict`
3. Maps every key from the HF naming convention (`transformer.h.0.attn.c_attn.weight`) to ours (`h.0.attn.c_attn.weight`)
4. Transposes the weight matrices for `c_attn`, `c_proj`, `c_fc`, and `mlp.c_proj` (the four Conv1D layers per block)
5. Copies embeddings and LayerNorm parameters directly (no transpose needed)
6. Skips `lm_head.weight` because it is weight-tied with `wte.weight`
7. Loads the mapped state dict with `strict=True` to ensure no keys are missing or unexpected

If `load_state_dict` succeeds with `strict=True`, every parameter in the model has been accounted for — there are no missing or extra keys.

## Training

```bash
# Stage 1: Pretrain on Dolma
python train_gpt2.py --stage pretrain

# Stage 2: Mid-training annealing
python train_gpt2.py --stage midtrain

# Stage 3: Supervised fine-tuning on Tulu-3
python train_gpt2.py --stage sft

# All stages sequentially
python train_gpt2.py --stage all

# Limit token budget (useful for testing)
python train_gpt2.py --max-tokens 10000000
```

## Evaluation

```bash
# Perplexity on WikiText-2
python evaluate_model.py

# Perplexity + text generation samples
python evaluate_model.py --generate

# Evaluate HuggingFace GPT-2 as a baseline
python evaluate_model.py --hf-model openai-community/gpt2

# Evaluate a specific checkpoint
python evaluate_model.py --checkpoint ./results/gpt2_results/pretrain/checkpoint-10000
```

## Upload to HuggingFace

```bash
python upload_to_huggingface.py \
  --repo-name gpt2-124m-dolma \
  --stage pretrain \
  --perplexity 29.4
```
