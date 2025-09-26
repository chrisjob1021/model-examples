  Key problems in your config:

  1. Gradient clipping too high: max_grad_norm=10.0 → reduce to 1.0-2.0
  2. Learning rate potentially unstable: initial_lr = 2e-4 with AdamW
  3. Min LR too high: min_lr_rate: 0.10 (10% of initial) prevents proper convergence
  4. Beta2 too low: adam_beta2=0.99 → should be 0.999 for stability
  5. CutMix disabled: Line 719 has it commented out

  Recommended changes:

  # Line 546-552 - Adjust learning rate
  initial_lr = 1e-4  # More stable for CNNs

  # Line 579 - Fix beta2
  adam_beta2=0.999,  # Standard value for stability

  # Line 646 - Tighter gradient clipping
  max_grad_norm=2.0,  # Prevent gradient explosions

  # Line 649 - Lower min LR
  "min_lr_rate": 0.01,  # 1% instead of 10%

  # Line 719 - Enable CutMix
  data_collator=cutmix_collator,  # Uncomment for regularization

  The spikes occur because:
  - High gradients (>4) cause parameter jumps
  - Beta2=0.99 makes optimizer too reactive to gradient changes
  - Min LR at 10% prevents proper convergence in cosine tail
  - No CutMix means less regularization
