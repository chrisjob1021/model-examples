# TODO: Fix CutMix + BatchNorm Eval Instability

## Problem
When CutMix is enabled, eval_loss shows instability/spikes while train_loss remains stable.
Root cause: BatchNorm running statistics are computed on mixed images during training,
but evaluation uses clean unmixed images. This distribution mismatch causes eval instability.

## Solution: Reset BatchNorm Stats Before Evaluation
Add `evaluation_loop()` override to CNNTrainer that resets BatchNorm running statistics
on clean eval data before each evaluation pass.

### Implementation Details
1. Override `evaluation_loop()` in CNNTrainer class (prelu_cnn.py:~950)
2. Before calling `super().evaluation_loop()`, call `_reset_batchnorm_stats(dataloader)`
3. In `_reset_batchnorm_stats()`:
   - Put model in train mode (to update running stats)
   - Reset all BatchNorm2d running stats via `module.reset_running_stats()`
   - Forward pass on ~100 batches of clean eval data to recompute stats
   - Restore original training mode
4. This ensures BatchNorm stats match clean image distribution during eval

### Benefits
- Eliminates eval_loss spikes when using CutMix
- No impact on training speed (only affects eval)
- Standard practice for CutMix + BatchNorm combinations
- Used in official CutMix paper implementation

### Alternative Solutions (if above doesn't work)
1. Use GroupNorm instead of BatchNorm (no running stats, but slower)
2. Use EMA (Exponential Moving Average) of model weights for evaluation
3. Disable CutMix for last N epochs before evaluation

### References
- CutMix paper: https://arxiv.org/abs/1905.04899
- Official implementation handles this issue

---

# TODO: Implement MixUp + CutMix Combined

## Problem
Currently only CutMix is implemented (and disabled due to BatchNorm issues).
MixUp is another effective augmentation technique that can be combined with CutMix
for better regularization and improved accuracy on ImageNet.

## What is MixUp?
MixUp blends entire images and their labels, unlike CutMix which pastes rectangular patches.
- Formula: `x_mixed = λ*x_i + (1-λ)*x_j` and `y_mixed = λ*y_i + (1-λ)*y_j`
- Simpler than CutMix (no spatial cutting)
- Often combined with CutMix in modern training recipes

## Solution: Implement Combined MixUp + CutMix Collator

### Implementation Details
1. **Create `MixUpCutMixCollator` class** (train_cnn_imagenet.py:~28)
   - Replace or extend existing `CutMixCollator`
   - Add MixUp implementation alongside CutMix
   - Randomly choose between MixUp and CutMix per batch

2. **Hyperparameters to add:**
   ```python
   mixup_alpha = 0.2      # Beta distribution parameter for MixUp
   mixup_prob = 0.25      # Probability of applying MixUp (25%)
   cutmix_prob = 0.25     # Probability of applying CutMix (25%)
   # Total augmentation: 50% of batches get either MixUp or CutMix
   ```

3. **Implementation logic:**
   ```python
   def __call__(self, batch):
       batch_dict = default_collate(batch)
       rand = random.random()

       if rand < self.mixup_prob:
           # Apply MixUp
           mixed_images, mixed_labels = self.mixup(...)
       elif rand < self.mixup_prob + self.cutmix_prob:
           # Apply CutMix
           mixed_images, mixed_labels = self.cutmix(...)
       else:
           # No augmentation (50% of batches)
           pass

       return batch_dict
   ```

4. **Use torchvision.transforms.v2:**
   - `T2.MixUp(num_classes=1000, alpha=mixup_alpha)`
   - `T2.CutMix(num_classes=1000, alpha=cutmix_alpha)`
   - Already available in torchvision, similar to existing CutMix implementation

### Standard Hyperparameters for ResNet-50 ImageNet

| Paper/Implementation | MixUp Alpha | MixUp Prob | CutMix Alpha | CutMix Prob |
|---------------------|-------------|------------|--------------|-------------|
| **Standard practice** | 0.2-0.8 | 0.25-0.5 | 1.0 | 0.25-0.5 |
| **timm defaults** | 0.8 | 0.5 | 1.0 | 0.5 |
| **Conservative (recommended start)** | 0.2 | 0.25 | 1.0 | 0.25 |

### Benefits
- **Complementary augmentation**: MixUp blends entire images, CutMix forces localization
- **Better regularization**: Combined use improves over either alone
- **Standard practice**: Most modern ImageNet recipes use both
- **Flexible**: Can tune probabilities independently

### Important Notes
- **Must fix BatchNorm issue first** (previous TODO item)
- MixUp has same BatchNorm distribution mismatch issue as CutMix
- The `_reset_batchnorm_stats()` solution works for both

### References
- MixUp paper: https://arxiv.org/abs/1710.09412
- CutMix paper: https://arxiv.org/abs/1905.04899
- timm implementation: Uses both MixUp and CutMix together
- ResNet-RS: Uses both for improved training

### Order of Implementation
1. First: Fix BatchNorm stats reset (previous TODO)
2. Second: Add MixUp alongside CutMix (this TODO)
3. Third: Enable both with conservative probabilities (0.25 each)
4. Fourth: Tune probabilities based on training stability
