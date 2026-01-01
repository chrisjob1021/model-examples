#!/usr/bin/env python3
"""Upload trained CNN model to HuggingFace Hub.

This script uploads a trained CNN checkpoint to the HuggingFace Hub,
including model weights, configuration, and a model card.
"""

import os
import sys
import json
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

# Add parent directory to path to import shared_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import find_latest_checkpoint
from prelu_cnn import CNN


def create_model_card(
    model_name: str,
    use_prelu: bool,
    num_classes: int,
    checkpoint_path: str,
    top1_acc: float = None,
    top5_acc: float = None,
) -> str:
    """Generate a model card README for the HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        Name of the model
    use_prelu : bool
        Whether the model uses PReLU activation
    num_classes : int
        Number of output classes
    checkpoint_path : str
        Path to the checkpoint
    top1_acc : float, optional
        Top-1 accuracy on validation set
    top5_acc : float, optional
        Top-5 accuracy on validation set

    Returns
    -------
    str
        Model card in markdown format
    """
    activation = "PReLU" if use_prelu else "ReLU"

    # Load training info if available
    trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
    training_info = ""
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            epoch = trainer_state.get('epoch', 'N/A')
            global_step = trainer_state.get('global_step', 'N/A')
            training_info = f"""
## Training Details

- **Epochs**: {epoch}
- **Global Steps**: {global_step:,}
- **Training Loss**: {trainer_state.get('best_metric', 'N/A')}
"""

    # Add accuracy info if provided
    accuracy_info = ""
    if top1_acc is not None or top5_acc is not None:
        accuracy_info = "\n## Evaluation Results\n\n"
        if top1_acc is not None:
            accuracy_info += f"- **Top-1 Accuracy**: {top1_acc:.2f}%\n"
        if top5_acc is not None:
            accuracy_info += f"- **Top-5 Accuracy**: {top5_acc:.2f}%\n"

        # Add comparison table if we have accuracy metrics
        if top1_acc is not None:
            top1_error = 100 - top1_acc
            top5_error = 100 - top5_acc if top5_acc else None

            accuracy_info += f"- **Top-1 Error**: {top1_error:.2f}%\n"
            if top5_error is not None:
                accuracy_info += f"- **Top-5 Error**: {top5_error:.2f}%\n"

            accuracy_info += "\n### Comparison with ImageNet Benchmarks\n\n"
            accuracy_info += "| Model | Top-1 | Top-5 | Parameters | Year | Notes |\n"
            accuracy_info += "|-------|-------|-------|------------|------|-------|\n"
            accuracy_info += "| AlexNet | 57.0% | 80.3% | 60M | 2012 | First deep CNN |\n"
            accuracy_info += "| VGG-16 | 71.5% | 90.1% | 138M | 2014 | Deep with small filters |\n"
            accuracy_info += "| **ResNet-50** | **76.0%** | **93.0%** | **25M** | **2015** | **Baseline** |\n"
            accuracy_info += "| ResNet-152 | 78.3% | 94.3% | 60M | 2015 | Deeper variant |\n"
            accuracy_info += "| Inception-v3 | 78.0% | 93.9% | 24M | 2015 | Multi-scale |\n"

            top5_str = f"{top5_acc:.2f}%" if top5_acc else "N/A"
            accuracy_info += f"| **This model** | **{top1_acc:.2f}%** | **{top5_str}** | **~23M** | **2025** | **{activation}** |\n"

            # Add achievement highlights if beating ResNet-50
            if top1_acc > 76.0:
                improvement = top1_acc - 76.0
                accuracy_info += f"\n**Key Achievement**: +{improvement:.2f}% improvement over ResNet-50 baseline\n"

    model_card = f"""---
license: mit
tags:
- image-classification
- pytorch
- cnn
- imagenet
- {activation.lower()}
datasets:
- imagenet-1k
---

# {model_name}

A Convolutional Neural Network (CNN) trained on ImageNet-1k with {activation} activation.

**Repository**: [github.com/chrisjob1021/model-examples](https://github.com/chrisjob1021/model-examples)

## Model Description

This is a ResNet-style CNN architecture featuring:
- **Activation Function**: {activation}
- **Number of Classes**: {num_classes}
- **Architecture**: Deep residual network with bottleneck blocks
- **Training Dataset**: ImageNet-1k
- **Code**: See `cnn/` directory in the repository for training scripts and model implementation

### Key Features

- Residual connections for better gradient flow
- Batch normalization for training stability
- {activation} activation for {'learnable non-linearity' if use_prelu else 'standard non-linearity'}
- Manual and builtin convolution implementations for educational purposes
{training_info}{accuracy_info}

## Usage

```python
from prelu_cnn import CNN

# Load the model
model = CNN.from_pretrained(
    "your-username/{model_name}",
    use_prelu={use_prelu},
    num_classes={num_classes}
)

# Use for inference
import torch
from torchvision import transforms
from PIL import Image

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("path/to/image.jpg")
input_tensor = transform(image).unsqueeze(0)

# Get prediction
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
```

## Training Procedure

This model was trained on ImageNet-1k using advanced techniques:

### Optimization
- **Optimizer**: AdamW (weight_decay=0.02)
- **Learning Rate**: 0.1 with 10-epoch warmup
- **Schedule**: Cosine annealing
- **Batch Size**: 1024 effective (512 per GPU × 2 gradient accumulation)
- **Mixed Precision**: fp16 for efficiency

### Data Augmentation
- **MixUp** (α=0.2, prob=50%): Linearly combines pairs of images and labels [[Zhang et al., 2017]](https://arxiv.org/abs/1710.09412)
- **CutMix** (α=1.0, prob=50%): Replaces image regions with patches from other images [[Yun et al., 2019]](https://arxiv.org/abs/1905.04899)
- **RandAugment** (magnitude=9, std=0.5): Automated augmentation policy [[Cubuk et al., 2020]](https://arxiv.org/abs/1909.13719)
- **Standard**: RandomResizedCrop (224×224), random horizontal flip, color jitter, random erasing (prob=0.25)

### Regularization
- Stochastic depth (drop_path_rate=0.1)
- Label smoothing (via MixUp/CutMix)
- Weight decay
- Batch normalization

## Model Architecture

### ResNet-50 with {activation}

```
CNN(
  conv1: [ConvAct(3 → 64, 7×7, stride=2) + MaxPool(3×3, stride=2)]
    Input: 224×224 → 112×112 → 56×56

  conv2_x: 3× BottleneckBlock(64 → 64 → 256)
    56×56 (no downsampling)

  conv3_x: 4× BottleneckBlock(256 → 128 → 512)
    56×56 → 28×28 (first block stride=2)

  conv4_x: 6× BottleneckBlock(512 → 256 → 1024)
    28×28 → 14×14 (first block stride=2)

  conv5_x: 3× BottleneckBlock(1024 → 512 → 2048)
    14×14 → 7×7 (first block stride=2)

  avgpool: AdaptiveAvgPool2d(1×1)
    7×7 → 1×1

  fc: Linear(2048 → {num_classes})
)
```

**Total Layers**: 50 (1 + 3×3 + 4×3 + 6×3 + 3×3 = 49 conv + 1 fc)

### Key Features

- **{activation} Activation**: {'Learnable negative slope for adaptive non-linearity' if use_prelu else 'Standard ReLU activation'}
- **Bottleneck Blocks**: 1×1 → 3×3 → 1×1 design (4× parameter reduction)
- **Residual Connections**: Skip connections for deep network training
- **ReZero Scaling**: Learnable residual scaling (initialized at 0)
- **Stochastic Depth**: Linear decay DropPath (0.0 → 0.1)
- **Batch Normalization**: Momentum=0.01 for stable statistics
- **Global Average Pooling**: Spatial invariance, zero parameters

### References

- **ResNet**: He et al., ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385), CVPR 2016
{'- **PReLU**: He et al., ["Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852), ICCV 2015' if use_prelu else ''}
- **Stochastic Depth**: Huang et al., ["Deep Networks with Stochastic Depth"](https://arxiv.org/abs/1603.09382), ECCV 2016
- **ReZero**: Bachlechner et al., ["ReZero is All You Need"](https://arxiv.org/abs/2003.04887), UAI 2021

## Citation

If you use this model, please cite:

```bibtex
@misc{{{model_name.replace('-', '_')},
  title={{{model_name}: CNN with {activation} for ImageNet Classification}},
  year={{2025}},
  publisher={{HuggingFace Hub}},
}}
```
"""

    # Add PReLU citation if applicable
    if use_prelu:
        model_card += """
### Original PReLU Paper

This model uses PReLU activation. Please also cite the original paper:

```bibtex
@inproceedings{he2015delving,
  title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1026--1034},
  year={2015}
}
```
"""

    model_card += """

## License

This model is released under the MIT License.
"""
    return model_card


def create_config(use_prelu: bool, num_classes: int, prelu_channel_wise: bool = True) -> dict:
    """Create model configuration dictionary.

    Parameters
    ----------
    use_prelu : bool
        Whether the model uses PReLU activation
    num_classes : int
        Number of output classes
    prelu_channel_wise : bool, optional
        Whether to use channel-wise PReLU

    Returns
    -------
    dict
        Model configuration
    """
    return {
        "model_type": "cnn",
        "architecture": "resnet-style",
        "use_prelu": use_prelu,
        "prelu_channel_wise": prelu_channel_wise,
        "num_classes": num_classes,
        "use_builtin_conv": True,
        "input_size": [3, 224, 224],
        "num_parameters": None,  # Will be filled in
    }


def upload_model_to_hub(
    checkpoint_path: str,
    repo_name: str,
    use_prelu: bool = True,
    num_classes: int = 1000,
    prelu_channel_wise: bool = True,
    organization: str = None,
    private: bool = False,
    top1_acc: float = None,
    top5_acc: float = None,
):
    """Upload a trained model to HuggingFace Hub.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint directory
    repo_name : str
        Name for the HuggingFace repository (e.g., "cnn-prelu-imagenet")
    use_prelu : bool, optional
        Whether the model uses PReLU activation
    num_classes : int, optional
        Number of output classes
    prelu_channel_wise : bool, optional
        Whether to use channel-wise PReLU
    organization : str, optional
        HuggingFace organization to upload to (defaults to user account)
    private : bool, optional
        Whether to make the repository private
    top1_acc : float, optional
        Top-1 accuracy for model card
    top5_acc : float, optional
        Top-5 accuracy for model card
    """
    print("=" * 60)
    print("🚀 Uploading Model to HuggingFace Hub")
    print("=" * 60)

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_path}")

    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📦 Repository: {repo_name}")
    if organization:
        print(f"🏢 Organization: {organization}")

    # Create temporary upload directory
    upload_dir = Path("./temp_upload")
    upload_dir.mkdir(exist_ok=True)

    try:
        # Copy model weights
        import shutil
        print(f"\n📋 Preparing upload files...")
        shutil.copy2(model_path, upload_dir / "model.safetensors")
        print(f"   ✅ Copied model weights")

        # Create and save config
        config = create_config(use_prelu, num_classes, prelu_channel_wise)

        # Load model to get parameter count
        print(f"\n🔧 Loading model to get metadata...")
        model = CNN(
            use_prelu=use_prelu,
            use_builtin_conv=True,
            prelu_channel_wise=prelu_channel_wise,
            num_classes=num_classes
        )
        config["num_parameters"] = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {config['num_parameters']:,}")

        with open(upload_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   ✅ Created config.json")

        # Create model card
        model_card = create_model_card(
            repo_name, use_prelu, num_classes, checkpoint_path, top1_acc, top5_acc
        )
        with open(upload_dir / "README.md", 'w') as f:
            f.write(model_card)
        print(f"   ✅ Created README.md")

        # Copy training state if available
        trainer_state_path = os.path.join(checkpoint_path, 'trainer_state.json')
        if os.path.exists(trainer_state_path):
            shutil.copy2(trainer_state_path, upload_dir / "trainer_state.json")
            print(f"   ✅ Copied trainer_state.json")

        # Initialize HuggingFace API
        print(f"\n🔐 Authenticating with HuggingFace...")
        api = HfApi()

        # Get user info
        user_info = api.whoami()
        username = user_info['name']
        print(f"   Logged in as: {username}")

        # Create repository name
        full_repo_name = f"{organization}/{repo_name}" if organization else f"{username}/{repo_name}"

        # Create repository
        print(f"\n📦 Creating repository: {full_repo_name}")
        try:
            create_repo(
                repo_id=full_repo_name,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            print(f"   ✅ Repository created/verified")
        except Exception as e:
            print(f"   ℹ️  Repository may already exist: {e}")

        # Upload files
        print(f"\n⬆️  Uploading files to HuggingFace Hub...")
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=full_repo_name,
            repo_type="model",
        )

        print(f"\n✅ Upload complete!")
        print(f"\n🌐 Model available at:")
        print(f"   https://huggingface.co/{full_repo_name}")

    finally:
        # Cleanup temporary directory
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            print(f"\n🧹 Cleaned up temporary files")


def main():
    """Main upload function."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload CNN model to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (defaults to latest in results/)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name for the HuggingFace repository (e.g., 'cnn-prelu-imagenet')"
    )
    parser.add_argument(
        "--use-prelu",
        action="store_true",
        default=None,
        help="Model uses PReLU activation (auto-detected if not specified)"
    )
    parser.add_argument(
        "--use-relu",
        action="store_true",
        help="Model uses ReLU activation (auto-detected if not specified)"
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="HuggingFace organization to upload to (optional)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--top1-acc",
        type=float,
        default=None,
        help="Top-1 accuracy to include in model card"
    )
    parser.add_argument(
        "--top5-acc",
        type=float,
        default=None,
        help="Top-5 accuracy to include in model card"
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Auto-detect from results directory
        results_dirs = [
            "./results/cnn_results_prelu",
            "./results/cnn_results_relu",
        ]

        checkpoint_path = None
        for results_dir in results_dirs:
            if os.path.exists(results_dir):
                found_checkpoint = find_latest_checkpoint(results_dir)
                if found_checkpoint:
                    checkpoint_path = found_checkpoint
                    break

        if not checkpoint_path:
            print("❌ No checkpoint found in results directories")
            print("   Please specify --checkpoint path")
            sys.exit(1)

    # Auto-detect activation type from checkpoint if not specified
    if args.use_prelu is None and not args.use_relu:
        from safetensors import safe_open
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(model_path):
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                use_prelu = any('.act.weight' in key or 'act1.weight' in key for key in keys)
        else:
            # Fallback to directory name
            use_prelu = 'prelu' in checkpoint_path.lower()
    else:
        use_prelu = args.use_prelu or not args.use_relu

    # Upload model
    upload_model_to_hub(
        checkpoint_path=checkpoint_path,
        repo_name=args.repo_name,
        use_prelu=use_prelu,
        num_classes=1000,
        organization=args.organization,
        private=args.private,
        top1_acc=args.top1_acc,
        top5_acc=args.top5_acc,
    )


if __name__ == "__main__":
    main()
