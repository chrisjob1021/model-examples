#!/usr/bin/env python3
"""Test script to verify CUDA device placement in CNN model"""

import torch
import numpy as np
import warnings
from prelu_cnn import CNN

def test_device_placement():
    """Test that all tensors are properly moved to the correct device."""
    
    # Capture warnings to ensure no manual implementation warnings during normal usage
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"GPU Memory Available: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        
        # Create model and move to device
        print("\n" + "="*50)
        print("Testing Model Device Placement")
        print("="*50)
        
        model = CNN(use_prelu=False, use_builtin_conv=True, num_classes=1000)
        model = model.to(device)
        
        # Check model parameters are on correct device
        print(f"Model is on device: {next(model.parameters()).device}")
        
        # Create dummy input
        batch_size = 2
        inputs = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 1000, (batch_size,))
        
        print(f"Input tensor device (before): {inputs.device}")
        print(f"Label tensor device (before): {labels.device}")
        
        # Move inputs to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        print(f"Input tensor device (after): {inputs.device}")
        print(f"Label tensor device (after): {labels.device}")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        model.eval()
        
        with torch.no_grad():
            # Hook to check intermediate tensor devices
            def check_device_hook(module, input, output):
                if hasattr(output, 'device'):
                    print(f"  {module.__class__.__name__}: input device={input[0].device}, output device={output.device}")
                elif isinstance(output, (list, tuple)):
                    print(f"  {module.__class__.__name__}: input device={input[0].device}, output devices={[o.device if hasattr(o, 'device') else 'N/A' for o in output]}")
            
            # Register hooks for key layers
            hooks = []
            for name, module in model.named_modules():
                if any(layer_type in str(type(module)) for layer_type in ['Conv', 'Pool', 'Linear', 'PReLU', 'ReLU']):
                    hook = module.register_forward_hook(check_device_hook)
                    hooks.append(hook)
            
            # Forward pass
            outputs = model(inputs)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        print(f"\nOutput tensor device: {outputs.device}")
        print(f"Output shape: {outputs.shape}")
        
        # Test backward pass
        print(f"\nTesting backward pass...")
        model.train()
        
        # Forward pass with gradients
        outputs = model(inputs)
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        print(f"Loss device: {loss.device}")
        print(f"Loss value: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients are on correct device
        grad_devices = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_devices.append((name, param.grad.device))
        
        print(f"\nGradient devices (sample):")
        for name, device in grad_devices[:5]:  # Show first 5
            print(f"  {name}: {device}")
        
        # Memory usage if CUDA
        if device.type == "cuda":
            print(f"\nGPU Memory after forward/backward:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Check for unexpected warnings
        manual_warnings = [warning for warning in w if 
                          "manual" in str(warning.message).lower() and 
                          "slow" in str(warning.message).lower()]
        
        if manual_warnings:
            error_msg = f"❌ UNEXPECTED WARNINGS during device placement test:\n"
            for warning in manual_warnings:
                error_msg += f"  - {warning.category.__name__}: {warning.message}\n"
            error_msg += "Manual implementation warnings should only appear during dedicated warning test!"
            raise RuntimeError(error_msg)
        
        print(f"\n✅ Device placement test completed successfully!")
        return True

def test_manual_layers():
    """Test that manual conv and pooling layers handle device correctly."""
    
    # Capture warnings to ensure no manual implementation warnings during normal usage
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n" + "="*50)
        print(f"Testing Manual Layers on {device}")
        print("="*50)
        
        from prelu_cnn import ManualConv2d, ManualMaxPool2d
        
        # Test ManualConv2d with builtin=True (should not warn)
        manual_conv = ManualConv2d(3, 64, kernel_size=7, stride=2, padding=3, use_builtin=True)
        manual_conv = manual_conv.to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"Input device: {x.device}")
        
        with torch.no_grad():
            out_conv = manual_conv(x)
            print(f"ManualConv2d output device: {out_conv.device}")
            print(f"ManualConv2d output shape: {out_conv.shape}")
        
        # Test ManualMaxPool2d with builtin=True (should not warn)
        manual_pool = ManualMaxPool2d(kernel_size=3, stride=3, use_builtin=True)
        
        with torch.no_grad():
            out_pool = manual_pool(out_conv)
            print(f"ManualMaxPool2d output device: {out_pool.device}")
            print(f"ManualMaxPool2d output shape: {out_pool.shape}")
        
        # Check for unexpected warnings
        manual_warnings = [warning for warning in w if 
                          "manual" in str(warning.message).lower() and 
                          "slow" in str(warning.message).lower()]
        
        if manual_warnings:
            error_msg = f"❌ UNEXPECTED WARNINGS during manual layers test:\n"
            for warning in manual_warnings:
                error_msg += f"  - {warning.category.__name__}: {warning.message}\n"
            error_msg += "Manual implementation warnings should only appear when use_builtin=False!"
            raise RuntimeError(error_msg)
        
        print(f"✅ Manual layers test completed successfully!")

def test_warnings():
    """Test that warnings are shown when manual implementations are used."""
    
    print(f"\n" + "="*50)
    print("Testing Warning Messages for Manual Implementations")
    print("="*50)
    
    from prelu_cnn import CNN, ManualConv2d, ManualMaxPool2d, ManualAdaptiveAvgPool2d
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        print("Creating CNN with manual convolutions (use_builtin_conv=False)...")
        model_manual = CNN(use_prelu=False, use_builtin_conv=False, num_classes=10)
        
        print("Creating ManualConv2d with manual implementation...")
        conv_manual = ManualConv2d(3, 64, kernel_size=3, use_builtin=False)
        
        print("Creating ManualMaxPool2d with manual implementation...")
        pool_manual = ManualMaxPool2d(kernel_size=2, use_builtin=False)
        
        print("Creating ManualAdaptiveAvgPool2d with manual implementation...")
        adaptive_pool_manual = ManualAdaptiveAvgPool2d((1, 1), use_builtin=False)
        
        # Test forward pass to trigger warnings
        x = torch.randn(1, 3, 32, 32)
        print("Running forward pass to trigger warnings...")
        conv_out = conv_manual(x)
        pool_out = pool_manual(conv_out)
        
        # Print all warnings
        print(f"\nWarnings captured ({len(w)} total):")
        for warning in w:
            print(f"  ⚠️  {warning.category.__name__}: {warning.message}")
        
        # Verify we got expected warnings
        manual_warnings = [warning for warning in w if 
                          "manual" in str(warning.message).lower() and 
                          "slow" in str(warning.message).lower()]
        
        if len(manual_warnings) < 2:  # Should have at least CNN + ManualConv2d warnings
            raise RuntimeError(f"❌ Expected manual implementation warnings, but only got {len(manual_warnings)}")
    
    print(f"\n✅ Warning test completed successfully!")

if __name__ == "__main__":
    test_device_placement()
    test_manual_layers() 
    test_warnings() 