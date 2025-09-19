# ImageNet Training Context (Cosine Scheduler Tweaks)

## What changed
- Added `get_cosine_with_hard_restarts_decay_schedule_with_warmup` in `cnn/prelu_cnn.py` to decay cosine restart peaks, add per-cycle warmup, and respect a floor via `min_lr_ratio`.
- `CNNTrainer.create_scheduler` now detects `cycle_decay`, `min_lr_ratio`, or `cycle_warmup_ratio` in `TrainingArguments.lr_scheduler_kwargs` and swaps in the custom scheduler.
- Default `TrainingArguments` in `cnn/train_cnn_imagenet.py` now pass:
  - `"num_cycles": 4`
  - `"cycle_decay": 0.55`
  - `"min_lr_ratio": 0.08`
  - `"cycle_warmup_ratio": 0.1`
  and we log the kwargs at startup for traceability.

## Why
- Original HF `cosine_with_restarts` jumped back to the base LR every restart, spiking loss.
- Custom scheduler keeps restarts but shrinks peak LR each cycle and eases into them with a 10% linear warmup.

## Current behavior expectations
- Cycle 0 peak = base LR, cycle 1 peak ≈ `base_lr * 0.55`, cycle 2 ≈ `base_lr * 0.55^2`, etc.
- LR ramps from `min_lr_ratio * base_lr` during the first 10% of every cycle before the cosine decay portion.

## What to check next
1. Run a short job (`python3 cnn/train_cnn_imagenet.py`) and open TensorBoard. LR curve should show decaying peaks and per-cycle warmup ramps.
2. Watch eval loss around restarts. If still unstable:
   - Increase `cycle_warmup_ratio` (e.g., 0.15) for softer ramps
   - Lower `cycle_decay` (<0.55) so peaks drop faster
   - Raise `min_lr_ratio` to keep floor higher if training stalls late.
3. Tune weight decay if validation overfits—remember larger batches need more explicit decay.

## Outstanding questions
- Do we need additional regularization once LR stabilizes? Monitor grad norms and weight norms.
- If cosine restarts are still too volatile, fall back to a plain cosine or piecewise schedule.

