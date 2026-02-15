"""GPT-2 (124M) implementation from scratch for Phase 1 of the transformer model card.

This module implements the full GPT-2 architecture as described in
"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

Architecture:
    - 12 transformer decoder layers
    - 768 hidden size, 12 attention heads (64 per head)
    - 3072 FFN inner dim (4x hidden)
    - 1024 token context window
    - 50,257 vocabulary (BPE)
    - Pre-LayerNorm, GELU activation, learned absolute positions
    - Bias in all linear layers

The implementation includes both a manual attention mechanism (educational)
and a PyTorch scaled_dot_product_attention path (production). Weights can be
loaded from the HuggingFace ``openai-community/gpt2`` checkpoint to validate
correctness before training from scratch.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GPT2Config:
    """Configuration for GPT-2 124M (Phase 1 defaults from MODEL_CARD.md)."""

    def __init__(
        self,
        vocab_size: int = 50_257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int = 3072,       # 4 * n_embd
        activation: str = "gelu",
        layer_norm_epsilon: float = 1e-5,
        bias: bool = True,
        dropout: float = 0.0,      # GPT-2 paper uses 0.1, but modern practice is 0.0 with other regularization
        tie_word_embeddings: bool = True,
        gradient_checkpointing: bool = False,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.bias = bias
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing

        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        self.head_dim = n_embd // n_head  # 64 for GPT-2


# ---------------------------------------------------------------------------
# Manual multi-head causal self-attention (educational)
# ---------------------------------------------------------------------------

class ManualCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit loops.

    This implementation makes the attention mechanism fully transparent:
    it splits Q/K/V by head, computes scaled dot-product attention with a
    causal mask, and concatenates the heads back together.

    Set ``use_builtin=True`` to delegate to ``F.scaled_dot_product_attention``
    for fused-kernel speed while keeping the same interface.
    """

    def __init__(self, config: GPT2Config, *, use_builtin: bool = True):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.use_builtin = use_builtin
        self.dropout = config.dropout

        # Combined QKV projection (matches HF GPT-2 ``c_attn``)
        # GPT-2 uses Conv1D which is just Linear with transposed weight storage.
        # We use standard nn.Linear — weight mapping handled at load time.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if not use_builtin:
            warnings.warn(
                "ManualCausalSelfAttention is using the explicit-loop path. "
                "Set use_builtin=True for fused-kernel speed.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project to Q, K, V  — shape (B, T, 3*C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_builtin:
            # PyTorch 2.0+ fused attention kernel — handles causal mask,
            # scaling, softmax, and dropout in a single CUDA call.
            # O(n) memory via FlashAttention-style tiling on supported hardware.
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # ---- Manual attention computation (educational) ----
            #
            # Scaled dot-product attention:
            #   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
            #
            # Scaling by 1/sqrt(d_k) prevents dot products from growing
            # with head dimension, which would push softmax into regions
            # with vanishingly small gradients.
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

            # Causal mask: prevent attending to future tokens.
            # Each position i can only attend to positions 0..i.
            # We fill future positions with -inf so softmax gives them 0 weight.
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            attn_out = torch.matmul(attn_weights, v)  # (B, n_head, T, head_dim)

        # Concatenate heads: (B, n_head, T, head_dim) → (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + residual dropout
        return self.resid_dropout(self.c_proj(attn_out))


# ---------------------------------------------------------------------------
# Feed-forward network (MLP)
# ---------------------------------------------------------------------------

class GPT2MLP(nn.Module):
    """Position-wise feed-forward network.

    GPT-2 FFN:  y = dropout(W_down(GELU(W_up(x))))
    Two linear layers with GELU activation in between.
    Inner dimension is 4x the embedding dimension (3072 for GPT-2).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        # Up projection (768 → 3072)
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=config.bias)
        # Down projection (3072 → 768)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU activation: smooth approximation of ReLU that allows small
        # negative gradients. Unlike ReLU, GELU has non-zero gradient
        # everywhere, which prevents dead neurons.
        #
        # GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
        # Intuitively: scale each element by the probability that a standard
        # normal exceeds it — large positive values pass through unchanged,
        # large negative values are zeroed, and values near zero get a smooth
        # transition between the two regimes.
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class GPT2Block(nn.Module):
    """Single transformer decoder block with Pre-LayerNorm.

    Pre-LayerNorm (GPT-2 style):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    This differs from the original transformer's Post-LayerNorm:
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + MLP(x))

    Pre-LayerNorm stabilizes training by normalizing inputs to each sublayer
    before the transformation, keeping the residual stream on a more uniform
    scale. This is critical for training deep transformers without careful
    learning rate warmup.
    """

    def __init__(self, config: GPT2Config, *, use_builtin_attn: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias)
        self.attn = ManualCausalSelfAttention(config, use_builtin=use_builtin_attn)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias)
        self.mlp = GPT2MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.ln_1(x))
        # Pre-norm MLP
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT-2 model
# ---------------------------------------------------------------------------

class GPT2(nn.Module):
    """GPT-2 decoder-only transformer language model.

    Architecture (124M):
        wte:  Token embedding        (vocab_size, n_embd) = (50257, 768)
        wpe:  Position embedding     (n_positions, n_embd) = (1024, 768)
        h:    12x GPT2Block
        ln_f: Final LayerNorm         (n_embd,) = (768,)
        lm_head: Linear(n_embd, vocab_size) = (768, 50257)  — weight-tied with wte

    wte (word token embedding): Lookup table of shape (vocab_size, n_embd):
    first dimension = vocabulary size (50257), second = embedding dimension (768).
    Token IDs index the first dimension to produce hidden vectors of size n_embd.
    The same matrix is reused as lm_head (output projection) via weight tying.

    Why lm_head uses the transpose: The shared matrix has shape (vocab_size, n_embd).
    For wte we index the first dimension (token_id → embedding vector). For the output
    we need a map (n_embd → vocab_size) to get logits. nn.Linear(n_embd, vocab_size)
    stores its weight as (out, in) = (vocab_size, n_embd) and computes output = input @ weight.T,
    so the same (vocab_size, n_embd) tensor is applied as the transpose, giving the
    (n_embd → vocab_size) map. One matrix thus serves both lookup and output projection.

    Weight tying: The language model head shares weights with the token
    embedding. This reduces parameters by ~38M (768*50257) and acts as a
    regularizer — the model must produce representations that are useful
    for both embedding tokens and predicting the next token.
    """

    def __init__(self, config: Optional[GPT2Config] = None, *, use_builtin_attn: bool = True):
        super().__init__()
        if config is None:
            config = GPT2Config()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.h = nn.ModuleList([
            GPT2Block(config, use_builtin_attn=use_builtin_attn)
            for _ in range(config.n_layer)
        ])

        # Final layer norm (applied after all blocks, before lm_head)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, bias=config.bias)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with token embedding
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to output projections
        # (residual stream accumulates contributions from each block)
        for block_idx, block in enumerate(self.h):
            # Scale down output projection weights by 1/sqrt(2*n_layer)
            # to prevent the residual stream variance from growing with depth.
            # Factor of 2 accounts for two residual additions per block (attn + mlp).
            scale = 1.0 / math.sqrt(2 * config.n_layer)
            with torch.no_grad():
                block.attn.c_proj.weight.mul_(scale)
                block.mlp.c_proj.weight.mul_(scale)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_no_embed = n_params - self.wpe.weight.numel()
        if config.tie_word_embeddings:
            # Don't double-count tied weights
            n_params_no_embed -= 0  # lm_head.weight is same tensor as wte.weight
        print(f"GPT-2 model initialized: {n_params:,} parameters ({n_params_no_embed:,} non-embedding)")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 conventions.

        - Linear layers: Normal(0, 0.02)
        - Embedding layers: Normal(0, 0.02)
        - LayerNorm: weight=1, bias=0
        - Biases: zero
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids : LongTensor of shape (B, T)
            Token indices.
        labels : LongTensor of shape (B, T), optional
            Target token indices for language modeling loss.
            Shifted internally so labels[i] = expected output for input_ids[i].

        Returns
        -------
        logits : FloatTensor of shape (B, T, vocab_size)
            If labels is None, returns logits only.
        loss : FloatTensor (scalar)
            If labels is provided, returns (loss, logits).
        """
        B, T = input_ids.size()
        assert T <= self.config.n_positions, (
            f"Sequence length {T} exceeds model context window {self.config.n_positions}"
        )

        # Position indices: 0, 1, 2, ..., T-1
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)

        # Token + position embeddings
        tok_emb = self.wte(input_ids)    # (B, T, n_embd)
        pos_emb = self.wpe(position_ids) # (1, T, n_embd) — broadcast over batch
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.h:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.h:
                x = block(x)

        # Final LayerNorm
        x = self.ln_f(x)

        # Language model head → logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if labels is not None:
            # Shift logits and labels for next-token prediction:
            #   logits[:, :-1, :] predicts labels[:, 1:]
            # This is the standard causal LM objective.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with torch.compile() prefix handling."""
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[GPT2Config] = None, device=None):
        """Load a trained GPT-2 model from a local checkpoint directory.

        Parameters
        ----------
        checkpoint_path : str
            Path to directory containing ``model.safetensors``.
        config : GPT2Config, optional
            Model configuration. Uses default 124M config if not provided.
        device : str or torch.device, optional
            Device to load the model on.

        Returns
        -------
        GPT2
            Model with loaded weights.
        """
        import os
        from safetensors.torch import load_file

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        print(f"Loading GPT-2 from checkpoint: {checkpoint_path}")

        if config is None:
            config = GPT2Config()

        model = cls(config)

        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")

        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        model = model.to(device)
        return model

    @classmethod
    def from_huggingface(cls, model_name: str = "openai-community/gpt2", device=None):
        """Load weights from a HuggingFace GPT-2 checkpoint to validate implementation.

        HuggingFace GPT-2 uses ``Conv1D`` layers whose weight matrices are stored
        as (in_features, out_features) — the transpose of ``nn.Linear``'s
        (out_features, in_features). This method handles that conversion.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier (default: ``openai-community/gpt2``).
        device : str or torch.device, optional
            Device to load the model on.

        Returns
        -------
        GPT2
            Model with HuggingFace weights loaded.
        """
        from transformers import GPT2LMHeadModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        print(f"Loading HuggingFace checkpoint: {model_name}")
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()

        config = GPT2Config(
            vocab_size=hf_model.config.vocab_size,
            n_positions=hf_model.config.n_positions,
            n_embd=hf_model.config.n_embd,
            n_layer=hf_model.config.n_layer,
            n_head=hf_model.config.n_head,
            n_inner=hf_model.config.n_inner or 4 * hf_model.config.n_embd,
            layer_norm_epsilon=hf_model.config.layer_norm_epsilon,
        )

        model = cls(config)

        # Build weight mapping: HF key → our key, with transpose flags
        # HF Conv1D stores weight as (in, out); nn.Linear stores (out, in).
        new_sd = {}

        # Embeddings (no transpose needed)
        new_sd["wte.weight"] = hf_sd["transformer.wte.weight"]
        new_sd["wpe.weight"] = hf_sd["transformer.wpe.weight"]

        # Final LayerNorm
        new_sd["ln_f.weight"] = hf_sd["transformer.ln_f.weight"]
        new_sd["ln_f.bias"] = hf_sd["transformer.ln_f.bias"]

        for i in range(config.n_layer):
            prefix_hf = f"transformer.h.{i}"
            prefix_ours = f"h.{i}"

            # Pre-attention LayerNorm
            new_sd[f"{prefix_ours}.ln_1.weight"] = hf_sd[f"{prefix_hf}.ln_1.weight"]
            new_sd[f"{prefix_ours}.ln_1.bias"] = hf_sd[f"{prefix_hf}.ln_1.bias"]

            # Attention QKV (Conv1D → Linear: transpose weight)
            new_sd[f"{prefix_ours}.attn.c_attn.weight"] = hf_sd[f"{prefix_hf}.attn.c_attn.weight"].t()
            new_sd[f"{prefix_ours}.attn.c_attn.bias"] = hf_sd[f"{prefix_hf}.attn.c_attn.bias"]

            # Attention output projection (Conv1D → Linear: transpose weight)
            new_sd[f"{prefix_ours}.attn.c_proj.weight"] = hf_sd[f"{prefix_hf}.attn.c_proj.weight"].t()
            new_sd[f"{prefix_ours}.attn.c_proj.bias"] = hf_sd[f"{prefix_hf}.attn.c_proj.bias"]

            # Pre-MLP LayerNorm
            new_sd[f"{prefix_ours}.ln_2.weight"] = hf_sd[f"{prefix_hf}.ln_2.weight"]
            new_sd[f"{prefix_ours}.ln_2.bias"] = hf_sd[f"{prefix_hf}.ln_2.bias"]

            # MLP up projection (Conv1D → Linear: transpose weight)
            new_sd[f"{prefix_ours}.mlp.c_fc.weight"] = hf_sd[f"{prefix_hf}.mlp.c_fc.weight"].t()
            new_sd[f"{prefix_ours}.mlp.c_fc.bias"] = hf_sd[f"{prefix_hf}.mlp.c_fc.bias"]

            # MLP down projection (Conv1D → Linear: transpose weight)
            new_sd[f"{prefix_ours}.mlp.c_proj.weight"] = hf_sd[f"{prefix_hf}.mlp.c_proj.weight"].t()
            new_sd[f"{prefix_ours}.mlp.c_proj.bias"] = hf_sd[f"{prefix_hf}.mlp.c_proj.bias"]

        # LM head — tied with wte in both HF and our implementation
        # HF stores it as ``lm_head.weight`` which equals ``transformer.wte.weight``
        # We tie them in __init__, so we don't load lm_head separately.
        # The tie means lm_head.weight IS wte.weight (same tensor).

        model.load_state_dict(new_sd, strict=True)
        model = model.to(device)

        print(f"Successfully loaded {model_name} weights into our GPT-2 implementation")
        return model

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """Autoregressive text generation.

        Parameters
        ----------
        input_ids : LongTensor of shape (B, T)
            Prompt token indices.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature (1.0 = unchanged, <1.0 = sharper, >1.0 = flatter).
        top_k : int, optional
            If set, only sample from the top-k most probable tokens.

        Returns
        -------
        LongTensor of shape (B, T + max_new_tokens)
            Generated token indices (prompt + continuation).
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to context window if sequence exceeds it
                idx_cond = input_ids if input_ids.size(1) <= self.config.n_positions else input_ids[:, -self.config.n_positions:]

                logits = self(idx_cond)
                # Take logits for the last position only
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ---------------------------------------------------------------------------
# Custom Trainer for causal language modeling
# ---------------------------------------------------------------------------

class GPT2Trainer(Trainer):
    """Custom HuggingFace Trainer for GPT-2 causal language modeling.

    Handles:
    - Causal LM loss computation (next-token prediction with label shifting)
    - Input validation and anomaly logging
    - TensorBoard logging of training diagnostics
    """

    def __init__(self, *args, error_log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_log_path = error_log_path
        self.gradient_logging_enabled = error_log_path is not None

        if self.gradient_logging_enabled:
            import os
            os.makedirs(os.path.dirname(error_log_path) if os.path.dirname(error_log_path) else ".", exist_ok=True)
            with open(error_log_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("GPT-2 TRAINING ANOMALY LOG\n")
                f.write("=" * 80 + "\n\n")

    def _log_anomaly(self, step, message, details=None):
        """Log anomaly to error log file."""
        if not self.gradient_logging_enabled:
            return
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.error_log_path, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"[{timestamp}] Step {step}: {message}\n")
            if details:
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute causal language modeling loss.

        The model's forward() handles the label shifting internally:
        logits[:, :-1] predicts labels[:, 1:].
        """
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", input_ids.clone())

        # Move to model device
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        labels = labels.to(model_device)

        # Check for input anomalies
        if hasattr(self, "state") and self.gradient_logging_enabled:
            if (input_ids < 0).any() or (input_ids >= model.config.vocab_size).any():
                self._log_anomaly(
                    self.state.global_step,
                    "Invalid token IDs detected",
                    {"min": input_ids.min().item(), "max": input_ids.max().item()},
                )

        loss, logits = model(input_ids=input_ids, labels=labels)

        # Check for loss anomalies
        if hasattr(self, "state") and self.gradient_logging_enabled:
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_anomaly(
                    self.state.global_step,
                    f"{'NaN' if torch.isnan(loss) else 'Inf'} loss detected",
                    {"loss": loss.item() if not torch.isnan(loss) else "NaN"},
                )

        if return_outputs:
            return loss, logits
        return loss
