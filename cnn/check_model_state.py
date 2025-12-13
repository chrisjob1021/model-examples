#!/usr/bin/env python3
"""Check model state to diagnose PReLU and ReZero issues"""

import torch
import torch.nn.functional as F
import os
from prelu_cnn import CNN
from shared_utils import find_latest_checkpoint

def main():
    print("=" * 80)
    print("MODEL STATE DIAGNOSTIC")
    print("=" * 80)

    # Find latest checkpoint
    base_dir = "./results/cnn_results_prelu"
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        print("Available directories in ./results:")
        if os.path.exists("./results"):
            for d in os.listdir("./results"):
                if os.path.isdir(os.path.join("./results", d)):
                    print(f"  - {d}")
        return

    checkpoint_path = find_latest_checkpoint(base_dir)

    if checkpoint_path is None:
        print(f"❌ No checkpoint found in {base_dir}")
        return

    print(f"\n📂 Loading checkpoint: {checkpoint_path}")

    # Create model (need to match the architecture used during training)
    model = CNN(
        use_prelu=True,  # We'll verify this
        use_builtin_conv=True,
        num_classes=1000,
        bn_momentum=0.1
    )

    # Load checkpoint - checkpoint_path is a directory, need to find the model file inside
    if os.path.isdir(checkpoint_path):
        # Look for model.safetensors or pytorch_model.bin in the checkpoint directory
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(model_path):
            print(f"   Loading model from: {model_path}")
            if model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
                # If it's a full checkpoint dict, extract just the state_dict
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print(f"✅ Loaded checkpoint successfully")
        else:
            raise FileNotFoundError(f"Could not find model.safetensors or pytorch_model.bin in {checkpoint_path}")
    else:
        # If it's a file, try loading it directly
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            # Assume it's just a state_dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint successfully")

    # Check 1: Verify activation types
    print("\n" + "=" * 80)
    print("CHECK 1: Activation Function Types")
    print("=" * 80)

    relu_count = 0
    prelu_count = 0

    for name, module in model.named_modules():
        if 'final_act' in name:
            module_type = type(module).__name__
            if 'ReLU' in module_type and 'PReLU' not in module_type:
                relu_count += 1
                print(f"❌ {name}: {module_type}")
            elif 'PReLU' in module_type:
                prelu_count += 1
                print(f"✅ {name}: {module_type}")

    print(f"\nSummary:")
    print(f"  ReLU layers: {relu_count}")
    print(f"  PReLU layers: {prelu_count}")

    if relu_count > 0:
        print(f"\n⚠️  WARNING: Model has ReLU activations (expected PReLU)!")
    else:
        print(f"\n✅ Model correctly uses PReLU activations")

    # Check 2: ReZero residual_scale parameters
    print("\n" + "=" * 80)
    print("CHECK 2: ReZero residual_scale Parameters")
    print("=" * 80)

    residual_scales = []
    for name, param in model.named_parameters():
        if 'residual_scale' in name:
            value = param.item()
            residual_scales.append((name, value))

    if not residual_scales:
        print("❌ No residual_scale parameters found!")
        print("   ReZero is NOT implemented in this checkpoint!")
    else:
        print(f"Found {len(residual_scales)} residual_scale parameters:")
        print(f"\n{'Block':<20} {'Raw Scale':<12} {'Softplus(Scale)':<18} {'Status':<25}")
        print("-" * 80)

        # Show all residual scales with softplus transformation
        for name, raw_value in residual_scales:
            # Extract block name (e.g., "conv2.0" from "conv2.0.residual_scale")
            block_name = name.replace('.residual_scale', '')

            # Apply softplus transformation (same as in model forward pass)
            scaled_value = F.softplus(torch.tensor(raw_value)).item()

            # Determine status
            if scaled_value < 0.1:
                status = "⚠️  VERY WEAK (< 0.1x)"
            elif scaled_value < 0.5:
                status = "⚠️  Weak (< 0.5x)"
            elif scaled_value < 1.0:
                status = "✓ Moderate (0.5-1x)"
            elif scaled_value < 2.0:
                status = "✓ Healthy (1-2x)"
            else:
                status = f"✓ Strong ({scaled_value:.1f}x)"

            print(f"{block_name:<20} {raw_value:<12.4f} {scaled_value:<18.4f} {status}")

        print("-" * 80)

        # Statistics
        raw_values = [v for _, v in residual_scales]
        scaled_values = [F.softplus(torch.tensor(v)).item() for v in raw_values]

        print(f"\nRaw Scale Statistics:")
        print(f"  Mean: {sum(raw_values)/len(raw_values):.6f}")
        print(f"  Min:  {min(raw_values):.6f}")
        print(f"  Max:  {max(raw_values):.6f}")

        print(f"\nActual Multiplier (Softplus) Statistics:")
        print(f"  Mean: {sum(scaled_values)/len(scaled_values):.4f}")
        print(f"  Min:  {min(scaled_values):.4f}")
        print(f"  Max:  {max(scaled_values):.4f}")

        # Check for issues
        weak_blocks = [(name.replace('.residual_scale', ''), v)
                       for name, v in residual_scales
                       if F.softplus(torch.tensor(v)).item() < 0.1]

        if weak_blocks:
            print(f"\n⚠️  WARNING: {len(weak_blocks)} blocks have very weak residuals (< 0.1x):")
            for block_name, raw_val in weak_blocks:
                scaled = F.softplus(torch.tensor(raw_val)).item()
                print(f"     {block_name}: softplus({raw_val:.4f}) = {scaled:.4f}")
            print(f"   These residual paths contribute < 10% of their magnitude!")
        elif min(scaled_values) < 0.5:
            print(f"\n⚠️  Some blocks have weak residuals (< 0.5x)")
        else:
            print(f"\n✅ All ReZero parameters show healthy residual contributions")

        print(f"\nInterpretation:")
        print(f"  - ReZero scales the residual path: out = shortcut + softplus(scale) * residual")
        print(f"  - Softplus ensures the scale is always positive")
        print(f"  - < 0.1x: Residual essentially dead (shortcut dominates)")
        print(f"  - 0.5-2x: Healthy balance between paths")
        print(f"  - > 2x: Residual dominates (less typical but can work)")
        print(f"  - If conv5 blocks show < 0.1x, this explains collapsed shortcuts")

    # Check 3: PReLU alpha parameters
    print("\n" + "=" * 80)
    print("CHECK 3: PReLU Alpha Parameters")
    print("=" * 80)

    prelu_params = []
    for name, param in model.named_parameters():
        if 'act' in name and 'weight' in name:
            prelu_params.append((name, param))

    if not prelu_params:
        print("❌ No PReLU alpha parameters found!")
        print("   Model likely uses ReLU, not PReLU!")
    else:
        print(f"Found {len(prelu_params)} PReLU layers:\n")

        # Sample a few
        for i, (name, param) in enumerate(prelu_params):
            if i < 3 or i >= len(prelu_params) - 3:
                mean_val = param.mean().item()
                min_val = param.min().item()
                max_val = param.max().item()
                print(f"  {name}:")
                print(f"    mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
            elif i == 3:
                print(f"  ... ({len(prelu_params) - 6} more) ...")

        # Overall statistics
        all_alphas = torch.cat([p.flatten() for _, p in prelu_params])
        print(f"\nOverall PReLU alpha statistics:")
        print(f"  Mean: {all_alphas.mean().item():.4f}")
        print(f"  Min:  {all_alphas.min().item():.4f}")
        print(f"  Max:  {all_alphas.max().item():.4f}")
        print(f"  Std:  {all_alphas.std().item():.4f}")

        # Check for issues
        if all_alphas.max().item() > 1.0:
            print(f"\n⚠️  WARNING: Some alpha values > 1.0 (amplifying negatives)")
        if all_alphas.min().item() < -0.5:
            print(f"\n⚠️  WARNING: Some alpha values < -0.5 (strong negative amplification)")

    # Check 4: Sample a few blocks to see complete structure
    print("\n" + "=" * 80)
    print("CHECK 4: Sample Block Structure")
    print("=" * 80)

    # Check conv5.0 specifically since it's problematic
    conv5_0_found = False
    for name, module in model.named_modules():
        if name == 'conv5.0':
            conv5_0_found = True
            print(f"\nBlock: {name}")
            print(f"  Type: {type(module).__name__}")

            # Check what final_act is
            if hasattr(module, 'final_act'):
                print(f"  final_act: {type(module.final_act).__name__}")

            # Check if residual_scale exists
            if hasattr(module, 'residual_scale'):
                print(f"  residual_scale: {module.residual_scale.item():.6f}")
            else:
                print(f"  residual_scale: NOT FOUND")

    if not conv5_0_found:
        print("❌ conv5.0 block not found in model!")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
