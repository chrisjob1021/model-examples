#!/usr/bin/env python3
"""Evaluate GPT-2 model on a validation set.

Calculates perplexity and average loss on held-out data.
Supports evaluation of any Phase 1 checkpoint (pretrain, midtrain, or sft).
"""

import os
import sys
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import find_latest_checkpoint
from gpt2 import GPT2, GPT2Config


def evaluate_perplexity(model, dataloader, device, verbose=True):
    """Evaluate model perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss).
    Lower is better. A perplexity of N means the model is as uncertain
    as a uniform distribution over N tokens, on average.

    Parameters
    ----------
    model : GPT2
        Trained model.
    dataloader : DataLoader
        Evaluation data loader.
    device : torch.device
        Device to run evaluation on.
    verbose : bool
        Whether to show progress bar.

    Returns
    -------
    tuple of (float, float)
        (perplexity, average_loss)
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    if verbose:
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", input_ids.clone()).to(device)

            loss, logits = model(input_ids=input_ids, labels=labels)

            # Count non-padding tokens (labels != -100)
            valid_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            if verbose:
                running_loss = total_loss / total_tokens if total_tokens > 0 else 0
                running_ppl = math.exp(min(running_loss, 100))  # Cap to avoid overflow
                pbar.set_postfix({
                    "Loss": f"{running_loss:.4f}",
                    "PPL": f"{running_ppl:.2f}",
                })

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(min(avg_loss, 100))

    return perplexity, avg_loss


def generate_samples(model, tokenizer, device, prompts=None, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate text samples for qualitative evaluation.

    Parameters
    ----------
    model : GPT2
        Trained model.
    tokenizer : AutoTokenizer
        Tokenizer.
    device : torch.device
        Device.
    prompts : list of str, optional
        Prompts to generate from. Uses defaults if None.
    max_new_tokens : int
        Tokens to generate per prompt.
    temperature : float
        Sampling temperature.
    top_k : int
        Top-k sampling parameter.
    """
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "In a galaxy far far away,",
            "def fibonacci(n):",
            "The quick brown fox",
            "Once upon a time there was",
        ]

    model.eval()
    print(f"\n{'=' * 60}")
    print("  Generated Samples")
    print(f"{'=' * 60}")

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--results-dir", type=str, default="./results/gpt2_results",
                        help="Results directory to search for checkpoints")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["pretrain", "midtrain", "sft"],
                        help="Which stage checkpoint to evaluate")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to evaluate on (default: wikitext-2)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of evaluation samples")
    parser.add_argument("--generate", action="store_true",
                        help="Generate text samples after evaluation")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="Evaluate HuggingFace GPT-2 as baseline (e.g., 'openai-community/gpt2')")
    args = parser.parse_args()

    print("=" * 60)
    print("  GPT-2 Model Evaluation")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if args.hf_model:
        print(f"\nLoading HuggingFace model: {args.hf_model}")
        model = GPT2.from_huggingface(args.hf_model, device=device)
    else:
        # Find checkpoint
        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            if args.stage:
                search_dir = os.path.join(args.results_dir, args.stage)
            else:
                # Try stages in reverse order (sft > midtrain > pretrain)
                for stage in ["sft", "midtrain", "pretrain"]:
                    search_dir = os.path.join(args.results_dir, stage)
                    if os.path.exists(search_dir):
                        break

            checkpoint_path = find_latest_checkpoint(search_dir)
            if not checkpoint_path:
                print(f"No checkpoint found in {search_dir}")
                print("Use --checkpoint to specify a path, or --hf-model for baseline")
                sys.exit(1)

        print(f"\nCheckpoint: {checkpoint_path}")
        model = GPT2.from_pretrained(checkpoint_path, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Load evaluation dataset
    dataset_name = args.dataset or "Salesforce/wikitext"
    print(f"\nLoading evaluation dataset: {dataset_name}")

    max_length = model.config.n_positions

    if dataset_name == "Salesforce/wikitext":
        raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raw = load_dataset(dataset_name, split="test", trust_remote_code=True)

    # Tokenize into fixed-length chunks
    def tokenize_fn(examples):
        all_ids = []
        for text in examples["text"]:
            if text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
                all_ids.append(tokenizer.eos_token_id)

        chunks = []
        for i in range(0, len(all_ids) - max_length, max_length):
            chunks.append(all_ids[i : i + max_length])

        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    eval_dataset = raw.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=raw.column_names,
        desc="Tokenizing eval data",
    )

    if args.max_samples and args.max_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(args.max_samples))

    eval_dataset.set_format(type="torch")
    print(f"Evaluation sequences: {len(eval_dataset):,} ({len(eval_dataset) * max_length:,} tokens)")

    # Create DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # Run evaluation
    print(f"\nRunning evaluation...")
    print("-" * 60)

    perplexity, avg_loss = evaluate_perplexity(model, dataloader, device)

    # Print results
    print(f"\n{'=' * 60}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Tokens Evaluated: {len(eval_dataset) * max_length:,}")
    print(f"{'=' * 60}")

    # Reference perplexities for context
    print(f"\nReference (WikiText-2 test, perplexity):")
    print(f"  GPT-2 (124M, HF):    ~29.4")
    print(f"  GPT-2 (355M, HF):    ~21.1")
    print(f"  GPT-2 (774M, HF):    ~17.5")
    print(f"  GPT-2 (1.5B, HF):    ~15.6")

    # Generate samples
    if args.generate:
        generate_samples(model, tokenizer, device)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
