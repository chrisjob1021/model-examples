# CNN Training Stability Improvements

## Problem Analysis
Based on TensorBoard metrics showing gradient norm spikes (~3.5) and evaluation loss instability, several issues were identified in the current training configuration.

## Key Issues and Solutions

### 1. Batch Size and Gradient Accumulation
**Problem**: Large batch size (1024) with high learning rate causing instability

**Why Smaller Batches Improve Stability:**

#### Gradient Noise and Stability
- **Gradient Variance**: Smaller batches have higher gradient variance (more noise), which helps:
  - Acts as implicit regularization
  - Helps escape sharp minima that generalize poorly
  - Prevents overfitting to specific batch patterns

#### Learning Rate Scaling Issues
- Large batches need proportionally higher learning rates (linear scaling rule)
- This rule breaks down beyond batch size ~2-4K
- Higher LRs with large batches → gradient explosions
- Spikes at batch=1024 suggest hitting this stability limit

#### Loss Landscape Navigation
- **Small batches (512)**: Noisy gradients explore loss landscape better, finding flatter minima
- **Large batches (1024+)**: Too-accurate gradients can get stuck in sharp minima, causing instability when trying to escape

#### Solution: Gradient Accumulation
Using batch=512 with grad_accum=2:
- Maintains effective batch of 1024 for throughput
- Computes gradients on smaller 512-sample batches
- Reduces per-step memory pressure and numerical errors

### 2. Gradient Clipping
**Current**: max_grad_norm=4.0
**Observed**: Gradients hovering around 3.5
**Recommended**: max_grad_norm=2.0
- Tighter clipping prevents explosive gradients
- Still allows sufficient gradient flow for learning

### 3. Learning Rate and Warmup
**Current Issues:**
- LR=2e-4 may be too aggressive
- Warmup ratio of 0.05 (5%) insufficient for stable start

**Recommended Changes:**
- Reduce initial LR to 1.5e-4
- Increase warmup_ratio to 0.1 (10%)
- This gives the optimizer more time to calibrate momentum buffers

### 4. AdamW Beta2 Parameter
**Current**: beta2=0.999 (very slow adaptation)
**Recommended**: beta2=0.99
- Makes optimizer more responsive to gradient changes
- Helps recover from instability events faster
- Standard value for vision tasks

### 5. Weight Decay
**Current**: weight_decay=0.05
**Analysis**: This is reasonable but could be tuned based on overfitting
- Monitor train vs eval loss gap
- If overfitting: increase to 0.1
- If underfitting: decrease to 0.01

## Implementation Changes

```python
# Before (unstable)
batch_size_per_gpu = 1024
grad_accum = 1
initial_lr = 2e-4
warmup_ratio = 0.05
max_grad_norm = 4
adam_beta2 = 0.999

# After (stable)
batch_size_per_gpu = 512
grad_accum = 2
initial_lr = 1.5e-4
warmup_ratio = 0.1
max_grad_norm = 2.0
adam_beta2 = 0.99
```

## Monitoring Guidelines

1. **Gradient Norms**: Should stay below 2.0 consistently
2. **Eval Loss**: Should decrease smoothly without spikes
3. **Train/Eval Gap**: Watch for overfitting (increasing gap)
4. **Learning Rate**: Monitor effective LR during warmup phase

## Additional Stability Techniques

### Mixed Precision Training
Consider adding for memory efficiency:
```python
training_args = TrainingArguments(
    ...,
    fp16=True,  # or bf16=True if available
    fp16_opt_level="O1",  # Conservative mixed precision
)
```

### Gradient Checkpointing
For very deep networks or memory constraints:
```python
model.gradient_checkpointing_enable()
```

### Learning Rate Scheduling
Current cosine schedule with min_lr=0.1 is good, but consider:
- Cosine with restarts for very long training
- Polynomial decay as alternative
- ReduceLROnPlateau if validation loss plateaus

## Expected Results

With these changes, you should see:
1. Gradient norms stabilizing below 2.0
2. Smoother evaluation loss curves
3. Better final accuracy
4. Faster convergence in early epochs
5. More robust training overall

## Emergency Fixes

If instability persists:
1. Further reduce batch_size to 256 with grad_accum=4
2. Reduce LR to 1e-4
3. Increase warmup to 20% of training
4. Try SGD with momentum=0.9 instead of AdamW
5. Check for data corruption or bad augmentation