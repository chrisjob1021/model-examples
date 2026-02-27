#!/usr/bin/env python3
"""Train GPT-2 (124M) using the 3-stage pipeline from MODEL_CARD.md Phase 1.

Stage 1: Pretraining on Dolma (web-scale language modeling)
Stage 2: Mid-training / Annealing on dolma3_dolmino_mix (high-quality data)
Stage 3: Post-training SFT on tulu-3-sft-mixture (instruction following)

Uses shared_utils.ModelTrainer for training orchestration and
shared_utils.find_latest_checkpoint for checkpoint management.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import ModelTrainer, find_latest_checkpoint
from gpt2 import GPT2, GPT2Config, GPT2Trainer


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def tokenize_pretrain(examples, tokenizer, max_length):
    """Tokenize pretraining text into fixed-length chunks.

    Concatenates all texts, then splits into non-overlapping chunks of
    ``max_length`` tokens. This maximizes GPU utilization by avoiding padding.
    """
    # Concatenate all texts with EOS separator
    all_ids = []
    encoded = tokenizer(examples["text"], add_special_tokens=False)["input_ids"]
    for ids in encoded:
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)

    # Split into fixed-length chunks (drop remainder > max_length)
    chunks = []
    for i in range(0, len(all_ids) - max_length, max_length):
        chunks.append(all_ids[i : i + max_length])

    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}


class StreamingPretrainDataset(IterableDataset):
    """Tokenizes a streaming HF dataset on-the-fly into fixed-length chunks.

    Yields ``{"input_ids": ..., "labels": ...}`` dicts of length
    ``max_length`` so training can start immediately without materializing
    the whole dataset.
    """

    def __init__(self, dataset_name, tokenizer, max_length, max_tokens, split="train"):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.split = split

    def __iter__(self):
        raw = load_dataset(
            self.dataset_name, split=self.split, streaming=True, trust_remote_code=True,
        )
        buffer = []
        tokens_yielded = 0
        batch_texts = []
        batch_size = 1000

        for example in raw:
            text = example.get("text", "")
            if not text.strip():
                continue
            batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                encoded = self.tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    buffer.extend(ids)
                    buffer.append(self.tokenizer.eos_token_id)
                batch_texts = []

                while len(buffer) >= self.max_length:
                    chunk = buffer[: self.max_length]
                    buffer = buffer[self.max_length :]
                    tokens_yielded += self.max_length
                    yield {"input_ids": chunk, "labels": chunk[:]}
                    if self.max_tokens and tokens_yielded >= self.max_tokens:
                        return

        # Flush: the main loop only tokenizes every 1000 texts, so up to 999
        # documents may remain in batch_texts, plus the buffer may hold enough
        # leftover tokens for more complete chunks. Drain both so we don't
        # silently drop the tail end of the data. Any remaining tokens shorter
        # than max_length are discarded (same as tokenize_pretrain).
        if batch_texts:
            encoded = self.tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                buffer.extend(ids)
                buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.max_length:
                chunk = buffer[: self.max_length]
                buffer = buffer[self.max_length :]
                tokens_yielded += self.max_length
                yield {"input_ids": chunk, "labels": chunk[:]}
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return


def tokenize_sft(examples, tokenizer, max_length):
    """Tokenize SFT examples in chat format.

    Formats multi-turn conversations into a single string using the
    tokenizer's chat template (if available) or a simple format.
    Only the assistant's tokens contribute to the loss; user/system
    tokens are masked with -100.
    """
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        # Build conversation text
        parts = []
        label_masks = []  # True = compute loss, False = mask with -100

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                text = f"<|system|>\n{content}\n"
                is_target = False
            elif role == "user":
                text = f"<|user|>\n{content}\n"
                is_target = False
            elif role == "assistant":
                text = f"<|assistant|>\n{content}<|endoftext|>\n"
                is_target = True
            else:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            parts.extend(tokens)
            label_masks.extend([is_target] * len(tokens))

        # Truncate to max_length
        parts = parts[:max_length]
        label_masks = label_masks[:max_length]

        # Build labels: -100 for masked positions (system/user), token IDs for assistant
        labels = [tok if mask else -100 for tok, mask in zip(parts, label_masks)]

        if len(parts) >= 16:  # Skip very short examples
            all_input_ids.append(parts)
            all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def build_streaming_eval_dataset(dataset_name, tokenizer, max_length, num_sequences=1000, split="train"):
    """Materialize a small eval set by streaming from the end of a dataset split."""
    from datasets import Dataset as HFDataset

    print(f"Building eval set ({num_sequences} sequences) from {dataset_name}...")
    raw = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    all_input_ids = []
    all_labels = []
    batch_texts = []

    for example in raw:
        text = example.get("text", "")
        if not text.strip():
            continue
        batch_texts.append(text)

        if len(batch_texts) >= 1000:
            result = tokenize_pretrain({"text": batch_texts}, tokenizer, max_length)
            all_input_ids.extend(result["input_ids"])
            all_labels.extend(result["labels"])
            batch_texts = []
            if len(all_input_ids) >= num_sequences:
                break

    if batch_texts:
        result = tokenize_pretrain({"text": batch_texts}, tokenizer, max_length)
        all_input_ids.extend(result["input_ids"])
        all_labels.extend(result["labels"])

    all_input_ids = all_input_ids[:num_sequences]
    all_labels = all_labels[:num_sequences]
    dataset = HFDataset.from_dict({"input_ids": all_input_ids, "labels": all_labels})
    print(f"Eval set ready: {len(dataset)} sequences")
    return dataset


def load_and_prepare_pretrain_dataset(
    dataset_name, tokenizer, max_length, max_tokens=None, split="train", streaming=False
):
    """Load a pretraining dataset and tokenize into fixed-length chunks.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
    tokenizer : AutoTokenizer
        Tokenizer for encoding text.
    max_length : int
        Context window size (1024 for GPT-2).
    max_tokens : int, optional
        Approximate upper bound on total tokens to use. If set, limits the
        number of raw examples loaded to stay within budget.
    split : str
        Dataset split to load.
    streaming : bool
        If True, use streaming mode for very large datasets.

    Returns
    -------
    Dataset
        Tokenized dataset with ``input_ids`` and ``labels`` columns.
    """
    print(f"Loading dataset: {dataset_name} (split={split})")

    if streaming:
        dataset = StreamingPretrainDataset(
            dataset_name, tokenizer, max_length, max_tokens, split=split,
        )
        total_tokens = max_tokens or 0
        print(f"Streaming dataset ready (up to ~{total_tokens:,} tokens)")
        return dataset
    else:
        raw = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if max_tokens:
            # Estimate examples needed: assume ~300 tokens per example on average
            avg_tokens_per_example = 300
            max_examples = int(max_tokens / avg_tokens_per_example) + 1
            if max_examples < len(raw):
                raw = raw.select(range(max_examples))
                print(f"  Limited to {max_examples:,} examples (~{max_tokens:,} target tokens)")

        dataset = raw.map(
            lambda examples: tokenize_pretrain(examples, tokenizer, max_length),
            batched=True,
            batch_size=1000,
            remove_columns=raw.column_names,
            num_proc=os.cpu_count(),
            desc="Tokenizing",
        )

    total_tokens = len(dataset) * max_length
    print(f"Dataset ready: {len(dataset):,} sequences ({total_tokens:,} tokens)")
    return dataset


def load_and_prepare_sft_dataset(dataset_name, tokenizer, max_length, split="train"):
    """Load and tokenize an SFT chat dataset.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
    tokenizer : AutoTokenizer
        Tokenizer for encoding text.
    max_length : int
        Maximum sequence length.
    split : str
        Dataset split.

    Returns
    -------
    Dataset
        Tokenized dataset with ``input_ids`` and ``labels`` columns.
    """
    print(f"Loading SFT dataset: {dataset_name} (split={split})")
    raw = load_dataset(dataset_name, split=split, trust_remote_code=True)

    dataset = raw.map(
        lambda examples: tokenize_sft(examples, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=raw.column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing SFT",
    )

    print(f"SFT dataset ready: {len(dataset):,} examples")
    return dataset


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage(
    stage_name,
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    tokenizer,
    num_epochs=1,
    max_steps=-1,
    learning_rate=6e-4,
    warmup_ratio=0.01,
    batch_size=8,
    grad_accum=4,
    lr_scheduler_type="cosine",
    use_bf16=False,
    use_fp16=False,
    weight_decay=0.1,
    disable_logging=False,
    resume_from_checkpoint=None,
):
    """Run a single training stage using ModelTrainer.

    Parameters
    ----------
    stage_name : str
        Human-readable name for logging.
    model : GPT2
        The model to train.
    train_dataset : Dataset
        Training data.
    eval_dataset : Dataset
        Evaluation data.
    output_dir : str
        Directory for checkpoints and logs.
    tokenizer : AutoTokenizer
        Tokenizer (for data collator padding).
    num_epochs : int
        Number of training epochs.
    max_steps : int
        Maximum training steps (-1 for epoch-based).
    learning_rate : float
        Peak learning rate.
    warmup_ratio : float
        Fraction of total steps used for warmup.
    batch_size : int
        Per-device batch size.
    grad_accum : int
        Gradient accumulation steps.
    lr_scheduler_type : str
        Learning rate schedule type.
    use_bf16 : bool
        Use BF16 mixed precision.
    use_fp16 : bool
        Use FP16 mixed precision.
    weight_decay : float
        AdamW weight decay.
    disable_logging : bool
        If True, skip TensorBoard logging.
    resume_from_checkpoint : str, optional
        Path to checkpoint to resume from.
    """
    print(f"\n{'=' * 60}")
    print(f"  STAGE: {stage_name}")
    print(f"{'=' * 60}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum,
        logging_steps=100,
        save_steps=2000,
        eval_steps=2000,
        seed=42,
        logging_dir=os.path.join(output_dir, "logs") if not disable_logging else None,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,  # GPT-3/Chinchilla convention for LLMs
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type=lr_scheduler_type,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        logging_strategy="steps",
        save_safetensors=False,  # torch.compile + weight tying causes shared memory error
        save_total_limit=3,
        load_best_model_at_end=False,
        prediction_loss_only=True,
        label_names=["labels"],
        report_to="tensorboard" if not disable_logging else "none",
    )

    # Data collator for padding variable-length sequences
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    error_log_path = os.path.join(output_dir, "anomalies.log")

    trainer = ModelTrainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer_class=GPT2Trainer,
        data_collator=data_collator,
        resume_from_checkpoint=resume_from_checkpoint,
        trainer_kwargs={"error_log_path": error_log_path},
    )

    # Log hyperparameters
    if not disable_logging and training_args.logging_dir:
        writer = SummaryWriter(log_dir=training_args.logging_dir)
        hparams = {
            "stage": stage_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "lr_scheduler": lr_scheduler_type,
        }
        import json
        writer.add_text("hyperparameters", f"```json\n{json.dumps(hparams, indent=2)}\n```", 0)
        writer.flush()
        writer.close()

    print(f"\nTraining configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Epochs: {num_epochs} (max_steps={max_steps})")
    print(f"  LR scheduler: {lr_scheduler_type}")
    print(f"  Mixed precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
    print(f"  Output: {output_dir}")
    print()

    hf_trainer, results = trainer.run()
    print(f"Stage '{stage_name}' completed. Results: {results}")

    return hf_trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 (124M) — Phase 1")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "midtrain", "sft", "all"],
                        help="Training stage to run (default: pretrain)")
    parser.add_argument("--no-logging", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Approximate token budget for pretraining (default: use full dataset)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--validate-weights", action="store_true",
                        help="Load HuggingFace GPT-2 weights to validate implementation, then exit")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    args = parser.parse_args()

    print("GPT-2 (124M) Training — Phase 1")
    print("=" * 50)

    # ---------------------------------------------------------------
    # Device setup
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Mixed precision
    use_bf16 = False
    use_fp16 = False
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Mixed precision: BF16")
        else:
            use_fp16 = True
            print("Mixed precision: FP16")

    # ---------------------------------------------------------------
    # Tokenizer — use GPT-2's BPE tokenizer
    # ---------------------------------------------------------------
    # Disable the tokenizer's built-in max length warning through a large model_max_length value
    # This is a hack to avoid the warning, and we handle chunking to max_length ourselves in tokenize_pretrain().
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", model_max_length=int(1e30))
    # GPT-2 tokenizer has no padding token; set it to EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    config = GPT2Config()
    max_length = config.n_positions  # 1024

    if args.validate_weights:
        print("\n--- Weight Validation Mode ---")
        from transformers import GPT2LMHeadModel

        # Compare on CPU for an exact float32 correctness check.
        #
        # Why not GPU? F.scaled_dot_product_attention dispatches to fused CUDA
        # kernels (FlashAttention / memory-efficient attention) that reorder
        # floating-point accumulations for speed — e.g. tiling the (T, T) attention
        # matrix into blocks and reducing partial sums in a different order than the
        # naïve loop. Because float addition is not associative ((a+b)+c != a+(b+c)
        # in finite precision), the fused result differs slightly from HF's
        # non-fused path, even with identical weights and inputs. These differences
        # (~1e-3 after 12 layers) are hardware-specific rounding artifacts, not
        # implementation bugs. On CPU both models use the same scalar reduction
        # order, so any remaining difference is a true architecture mismatch.
        compare_device = torch.device("cpu")
        print(f"Comparing logits on: {compare_device} (deterministic float32)")

        hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(compare_device)
        hf_model.eval()
        model = GPT2.from_huggingface("openai-community/gpt2", device=compare_device)
        model.eval()

        # Compare logits on a test input
        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(compare_device)

        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits
            our_logits = model(input_ids)

        max_diff = (hf_logits - our_logits).abs().max().item()
        mean_diff = (hf_logits - our_logits).abs().mean().item()
        print(f"\nLogit comparison (prompt: '{prompt}'):")
        print(f"  Max absolute difference:  {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        atol = 1e-4
        if torch.allclose(hf_logits, our_logits, atol=atol):
            print(f"  PASS: logits match within atol={atol}")
        else:
            print(f"  FAIL: logits differ beyond atol={atol}")
            return

        # Also generate text as a qualitative check (on the original device for speed)
        model = model.to(device)
        input_ids = input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated_text}")
        print("\nWeight validation successful!")
        return

    print(f"\nCreating GPT-2 model (124M)...")
    model = GPT2(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # torch.compile for speedup
    use_torch_compile = True
    if use_torch_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="default")
        print("Model compiled successfully")

    # ---------------------------------------------------------------
    # Checkpoint resume
    # ---------------------------------------------------------------
    base_output_dir = "./results/gpt2_results"
    checkpoint_path = None
    if args.resume:
        checkpoint_path = find_latest_checkpoint(base_output_dir)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoint found in {base_output_dir}, starting fresh")

    # ---------------------------------------------------------------
    # Stage 1: Pretraining
    # ---------------------------------------------------------------
    if args.stage in ("pretrain", "all"):
        # Dataset: allenai/dolma — 3T+ tokens, sample ~10-20B for 124M model
        # For feasible training, start with a smaller subset (controlled by --max-tokens)
        pretrain_dataset_name = "allenai/dolma3_pool"
        default_pretrain_tokens = 10_000_000_000  # 10B tokens (~80 tokens/param, ~24h on 1x L40S)

        max_tokens = args.max_tokens or default_pretrain_tokens

        train_dataset = load_and_prepare_pretrain_dataset(
            pretrain_dataset_name, tokenizer, max_length,
            max_tokens=max_tokens,
            split="train",
            streaming=args.streaming,
        )

        is_streaming = isinstance(train_dataset, IterableDataset)
        if is_streaming:
            eval_dataset = build_streaming_eval_dataset(
                pretrain_dataset_name, tokenizer, max_length,
            )
            # max_steps = total_tokens / (batch_size * grad_accum * max_length)
            pretrain_max_steps = int(max_tokens / (args.batch_size * args.grad_accum * max_length))
        else:
            eval_size = min(1000, len(train_dataset))
            eval_dataset = train_dataset.select(range(eval_size))
            pretrain_max_steps = -1

        # GPT-2 training hyperparameters (from Chinchilla/GPT-3 conventions)
        #
        # Learning rate: 6e-4 is the standard for 124M parameter models.
        # The scaling law from Kaplan et al. (2020) suggests:
        #   lr ~ 0.003239 * N^{-0.3} where N is parameter count
        #   For N=124M: lr ~ 0.003239 * 124e6^{-0.3} ~ 6e-4
        #
        # Weight decay: 0.1 is standard for transformer pretraining.
        # Higher than CNN convention (0.01-0.05) because transformers have
        # more parameters and benefit from stronger regularization.
        #
        # Warmup: ~1-2% of total steps. Prevents early training instability
        # when Adam's running statistics haven't converged yet.
        run_stage(
            stage_name="Stage 1: Pretraining (Dolma)",
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=os.path.join(base_output_dir, "pretrain"),
            tokenizer=tokenizer,
            num_epochs=1,
            max_steps=pretrain_max_steps,
            learning_rate=6e-4,
            warmup_ratio=0.01,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr_scheduler_type="cosine",
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            weight_decay=0.1,
            disable_logging=args.no_logging,
            resume_from_checkpoint=checkpoint_path,
        )

    # ---------------------------------------------------------------
    # Stage 2: Mid-training / Annealing
    # ---------------------------------------------------------------
    if args.stage in ("midtrain", "all"):
        midtrain_dataset_name = "allenai/dolma3_dolmino_mix-10B-1025"

        # For 124M model, use ~1-2B tokens for mid-training
        midtrain_max_tokens = args.max_tokens or 1_000_000_000

        train_dataset = load_and_prepare_pretrain_dataset(
            midtrain_dataset_name, tokenizer, max_length,
            max_tokens=midtrain_max_tokens,
            split="train",
            streaming=args.streaming,
        )

        is_streaming = isinstance(train_dataset, IterableDataset)
        if is_streaming:
            eval_dataset = build_streaming_eval_dataset(
                midtrain_dataset_name, tokenizer, max_length,
            )
            midtrain_max_steps = int(midtrain_max_tokens / (args.batch_size * args.grad_accum * max_length))
        else:
            eval_size = min(1000, len(train_dataset))
            eval_dataset = train_dataset.select(range(eval_size))
            midtrain_max_steps = -1

        # Mid-training uses decaying LR toward 0 (cosine to minimum)
        # Lower LR than pretraining since model is already trained
        if args.stage == "all":
            # Load from pretrain checkpoint
            pretrain_ckpt = find_latest_checkpoint(os.path.join(base_output_dir, "pretrain"))
            if pretrain_ckpt:
                print(f"Loading pretrained model from: {pretrain_ckpt}")
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(pretrain_ckpt, "model.safetensors"))
                model.load_state_dict(state_dict)
                print("Pretrained weights loaded")

        run_stage(
            stage_name="Stage 2: Mid-training (Annealing)",
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=os.path.join(base_output_dir, "midtrain"),
            tokenizer=tokenizer,
            num_epochs=1,
            max_steps=midtrain_max_steps,
            learning_rate=1e-4,  # Lower LR for annealing
            warmup_ratio=0.005,  # Shorter warmup
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr_scheduler_type="cosine",
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            weight_decay=0.1,
            disable_logging=args.no_logging,
        )

    # ---------------------------------------------------------------
    # Stage 3: Post-training SFT
    # ---------------------------------------------------------------
    if args.stage in ("sft", "all"):
        sft_dataset_name = "allenai/tulu-3-sft-mixture"

        train_dataset = load_and_prepare_sft_dataset(
            sft_dataset_name, tokenizer, max_length, split="train"
        )

        eval_size = min(1000, len(train_dataset))
        eval_dataset = train_dataset.select(range(eval_size))

        # SFT uses even lower LR and shorter training
        if args.stage == "all":
            midtrain_ckpt = find_latest_checkpoint(os.path.join(base_output_dir, "midtrain"))
            if midtrain_ckpt:
                print(f"Loading mid-trained model from: {midtrain_ckpt}")
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(midtrain_ckpt, "model.safetensors"))
                model.load_state_dict(state_dict)
                print("Mid-trained weights loaded")

        run_stage(
            stage_name="Stage 3: Post-training SFT (Tulu-3)",
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=os.path.join(base_output_dir, "sft"),
            tokenizer=tokenizer,
            num_epochs=2,  # SFT typically runs 2-3 epochs
            learning_rate=2e-5,  # Much lower LR for fine-tuning
            warmup_ratio=0.03,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr_scheduler_type="cosine",
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            weight_decay=0.01,  # Lower weight decay for SFT
            disable_logging=args.no_logging,
        )

    print(f"\nTraining completed!")
    print(f"Results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
