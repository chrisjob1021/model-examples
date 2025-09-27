## Gradient and Eval Loss Spikes Notes

- `train/grad_norm` on TensorBoard shows the raw gradient norm before clipping. The training script clamps gradients to a maximum of 4 (`TrainingArguments.max_grad_norm`). Spikes in the graph simply mean clipping activated.
- The spikes correspond to occasional outlier batches. With `per_device_train_batch_size=1024` and no gradient accumulation, a single hard batch can produce a large pre-clip norm. After clipping, optimization keeps going, which is why training does not diverge.
- Those same extreme batches nudge the weights off their typical path just before evaluation runs. As a result, `eval/loss` jumps at the epoch boundary, then settles as training continues.
- Augmentation intensity is the main driver of these outliers. The pipeline chains `RandomResizedCrop(scale=(0.08, 1.0))`, `RandAugment` (magnitude 7), and `RandomErasing(p=0.1)`, so an unlucky combination (tiny crop + strong color warp + erasing) can make an example very different from the validation distribution.

### Mitigation Ideas

1. Lower augmentation strength: raise the crop scale floor (e.g., 0.15–0.2), reduce `RandAugment` magnitude to 5, or move `RandomErasing` before normalization.
2. Add extra safety: tighten `max_grad_norm` to ~2–3 or reduce `initial_lr` slightly to soften the effect of clipped steps.
3. Stabilize batches: re-enable the CutMix collator (`data_collator=cutmix_collator`) or enable light gradient accumulation (`gradient_accumulation_steps=2`) so updates average over more examples.

Let me know which direction you want to try first, and I can prep the diff.
