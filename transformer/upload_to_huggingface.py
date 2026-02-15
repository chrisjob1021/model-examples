#!/usr/bin/env python3
"""Upload trained GPT-2 model to HuggingFace Hub.

Uploads model weights, configuration, and a model card.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import find_latest_checkpoint
from gpt2 import GPT2, GPT2Config


def create_model_card(
    model_name: str,
    config: GPT2Config,
    checkpoint_path: str,
    stage: str = "pretrain",
    perplexity: float = None,
    avg_loss: float = None,
) -> str:
    """Generate a model card for the HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Name of the model.
    config : GPT2Config
        Model configuration.
    checkpoint_path : str
        Path to the checkpoint.
    stage : str
        Training stage (pretrain, midtrain, sft).
    perplexity : float, optional
        Evaluation perplexity.
    avg_loss : float, optional
        Evaluation average loss.

    Returns
    -------
    str
        Model card in markdown format.
    """
    stage_descriptions = {
        "pretrain": "Pretraining on Dolma (web-scale language modeling)",
        "midtrain": "Mid-training / Annealing on high-quality data mix",
        "sft": "Supervised Fine-Tuning on Tulu-3 instruction mixture",
    }
    stage_desc = stage_descriptions.get(stage, stage)

    # Load training state if available
    training_info = ""
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            state = json.load(f)
            training_info = f"""
## Training Details

- **Epoch**: {state.get('epoch', 'N/A')}
- **Global Steps**: {state.get('global_step', 'N/A'):,}
- **Training Loss**: {state.get('best_metric', 'N/A')}
"""

    # Evaluation results
    eval_info = ""
    if perplexity is not None or avg_loss is not None:
        eval_info = "\n## Evaluation Results\n\n"
        if perplexity is not None:
            eval_info += f"- **Perplexity (WikiText-2)**: {perplexity:.2f}\n"
        if avg_loss is not None:
            eval_info += f"- **Average Loss**: {avg_loss:.4f}\n"

        eval_info += """
### Reference Perplexities (WikiText-2 test)

| Model | Parameters | Perplexity |
|-------|-----------|------------|
| GPT-2 (HF) | 124M | ~29.4 |
| GPT-2 Medium (HF) | 355M | ~21.1 |
| GPT-2 Large (HF) | 774M | ~17.5 |
| GPT-2 XL (HF) | 1.5B | ~15.6 |
"""
        if perplexity is not None:
            eval_info += f"| **This model** | **124M** | **{perplexity:.2f}** |\n"

    model_card = f"""---
license: mit
tags:
- text-generation
- pytorch
- gpt2
- transformer
- causal-lm
language:
- en
---

# {model_name}

A GPT-2 (124M) language model implemented from scratch and trained using a 3-stage pipeline.

**Repository**: [github.com/chrisjob1021/model-examples](https://github.com/chrisjob1021/model-examples)

## Model Description

This is a from-scratch GPT-2 implementation trained as Phase 1 of the transformer model card.
Current training stage: **{stage_desc}**

### Architecture

| Parameter | Value |
|-----------|-------|
| Layers | {config.n_layer} |
| Hidden size | {config.n_embd} |
| Attention heads | {config.n_head} |
| Head dimension | {config.head_dim} |
| FFN inner dim | {config.n_inner} |
| Context length | {config.n_positions} |
| Vocab size | {config.vocab_size:,} |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Position encoding | Learned absolute |
| Parameters | ~124M |

### Key Features

- From-scratch implementation (not a HuggingFace wrapper)
- Manual attention implementation for educational transparency
- Weight-compatible with `openai-community/gpt2` for validation
- 3-stage training pipeline: pretraining, mid-training, SFT
{training_info}{eval_info}

## Usage

```python
from gpt2 import GPT2, GPT2Config
from transformers import AutoTokenizer

# Load model
model = GPT2.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Generate text
import torch
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=40)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Training Pipeline (Phase 1)

### Stage 1: Pretraining
- **Dataset**: [allenai/dolma](https://hf.co/datasets/allenai/dolma) (ODC-BY)
- **Objective**: Autoregressive next-token prediction
- **LR**: 6e-4 with cosine decay

### Stage 2: Mid-training (Annealing)
- **Dataset**: [allenai/dolma3_dolmino_mix-10B-1025](https://hf.co/datasets/allenai/dolma3_dolmino_mix-10B-1025) (ODC-BY)
- **Content**: Math, code, science, high-quality web
- **LR**: 1e-4, cosine decay to 0

### Stage 3: SFT (Instruction Following)
- **Dataset**: [allenai/tulu-3-sft-mixture](https://hf.co/datasets/allenai/tulu-3-sft-mixture) (ODC-BY)
- **Format**: Multi-turn chat (system/user/assistant)
- **LR**: 2e-5

### References

- **GPT-2**: Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019
- **Chinchilla**: Hoffmann et al., ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556), 2022
- **Dolma**: Soldaini et al., ["Dolma: an Open Corpus of Three Trillion Tokens"](https://arxiv.org/abs/2402.00159), 2024

## Citation

```bibtex
@misc{{{model_name.replace('-', '_')},
  title={{{model_name}: GPT-2 124M from Scratch}},
  year={{2025}},
  publisher={{HuggingFace Hub}},
}}
```

## License

This model is released under the MIT License.
"""

    return model_card


def create_config_json(config: GPT2Config) -> dict:
    """Create model configuration dictionary."""
    return {
        "model_type": "gpt2",
        "architecture": "decoder-only-transformer",
        "vocab_size": config.vocab_size,
        "n_positions": config.n_positions,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_inner": config.n_inner,
        "activation": config.activation,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "bias": config.bias,
        "tie_word_embeddings": config.tie_word_embeddings,
    }


def upload_model(
    checkpoint_path: str,
    repo_name: str,
    stage: str = "pretrain",
    organization: str = None,
    private: bool = False,
    perplexity: float = None,
    avg_loss: float = None,
):
    """Upload model to HuggingFace Hub.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint directory.
    repo_name : str
        Repository name.
    stage : str
        Training stage.
    organization : str, optional
        HuggingFace organization.
    private : bool
        Whether to make repo private.
    perplexity : float, optional
        Eval perplexity for model card.
    avg_loss : float, optional
        Eval loss for model card.
    """
    print("=" * 60)
    print("  Uploading GPT-2 Model to HuggingFace Hub")
    print("=" * 60)

    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Repository: {repo_name}")

    upload_dir = Path("./temp_upload_gpt2")
    upload_dir.mkdir(exist_ok=True)

    try:
        # Copy model weights
        print("Preparing upload files...")
        shutil.copy2(model_path, upload_dir / "model.safetensors")

        # Create config
        config = GPT2Config()
        config_dict = create_config_json(config)

        # Get parameter count
        model = GPT2(config)
        config_dict["num_parameters"] = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {config_dict['num_parameters']:,}")

        with open(upload_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create model card
        model_card = create_model_card(
            repo_name, config, checkpoint_path, stage, perplexity, avg_loss
        )
        with open(upload_dir / "README.md", "w") as f:
            f.write(model_card)

        # Copy trainer state
        trainer_state = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state):
            shutil.copy2(trainer_state, upload_dir / "trainer_state.json")

        # Upload
        api = HfApi()
        user_info = api.whoami()
        username = user_info["name"]
        full_repo = f"{organization}/{repo_name}" if organization else f"{username}/{repo_name}"

        print(f"Creating repository: {full_repo}")
        create_repo(repo_id=full_repo, repo_type="model", private=private, exist_ok=True)

        print("Uploading files...")
        api.upload_folder(folder_path=str(upload_dir), repo_id=full_repo, repo_type="model")

        print(f"\nUpload complete!")
        print(f"Model: https://huggingface.co/{full_repo}")

    finally:
        if upload_dir.exists():
            shutil.rmtree(upload_dir)


def main():
    parser = argparse.ArgumentParser(description="Upload GPT-2 model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--repo-name", type=str, required=True,
                        help="HuggingFace repository name")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "midtrain", "sft"],
                        help="Training stage of the checkpoint")
    parser.add_argument("--organization", type=str, default=None,
                        help="HuggingFace organization")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private")
    parser.add_argument("--perplexity", type=float, default=None,
                        help="Evaluation perplexity for model card")
    parser.add_argument("--avg-loss", type=float, default=None,
                        help="Evaluation loss for model card")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        results_dir = f"./results/gpt2_results/{args.stage}"
        checkpoint_path = find_latest_checkpoint(results_dir)
        if not checkpoint_path:
            print(f"No checkpoint found in {results_dir}")
            sys.exit(1)

    upload_model(
        checkpoint_path=checkpoint_path,
        repo_name=args.repo_name,
        stage=args.stage,
        organization=args.organization,
        private=args.private,
        perplexity=args.perplexity,
        avg_loss=args.avg_loss,
    )


if __name__ == "__main__":
    main()
