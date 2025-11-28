# grad_monitor.py
"""Gradient histogram monitoring for CNN training.

Registers backward hooks on model parameters to track gradient statistics
and histograms over time. Useful for debugging training dynamics.
"""

import torch
import time
import csv
import math
from collections import defaultdict


class GradHistogram:
    """Monitor gradient histograms during training.

    Registers backward hooks on model parameters to capture gradient statistics
    and histograms at each backward pass.

    Usage:
        model = CNN(...)
        monitor = GradHistogram(
            model,
            modules_filter=lambda m, n: any(k in n for k in ["conv", "layer", "block"]),
            bins=41,
            track_layers=("weight",)
        )

        for step, (x, y) in enumerate(loader):
            loss = model(x).cross_entropy(y)
            loss.backward()
            optimizer.step()

            if (step + 1) % 200 == 0:
                monitor.export_csv(f"grad_hist_step_{step+1}.csv")

        monitor.close()
    """

    def __init__(
        self,
        model,
        modules_filter=None,  # None = all modules
        bins=41,
        track_layers=("weight",),
        log_every_n_steps=1,
    ):
        """Initialize gradient histogram monitor.

        Args:
            model: PyTorch model to monitor
            modules_filter: Function(module, name) -> bool to filter which modules to track.
                           Default None = track ALL modules.
            bins: Number of histogram bins
            track_layers: Tuple of parameter names to track (e.g., ("weight",))
            log_every_n_steps: Only record gradients every N backward passes
        """
        self.records = []
        self.bins = bins
        self.edges = None
        self.handles = []
        self.track_layers = track_layers
        self.log_every_n_steps = log_every_n_steps
        self._step_counter = 0
        self._enabled = True

        if modules_filter is None:
            modules_filter = lambda m, n: True  # Track all modules

        for name, mod in model.named_modules():
            if name == "" or not modules_filter(mod, name):
                continue
            for p_name, p in mod.named_parameters(recurse=False):
                if p_name not in track_layers or not p.requires_grad:
                    continue
                full_name = f"{name}.{p_name}"
                handle = p.register_hook(self._make_hook(full_name))
                self.handles.append(handle)

        print(f"📊 GradHistogram: Registered {len(self.handles)} hooks (all layers, weights only)")

    def _make_hook(self, pname):
        """Create a backward hook for a specific parameter."""
        def hook(grad):
            if not self._enabled:
                return grad

            self._step_counter += 1
            if self._step_counter % self.log_every_n_steps != 0:
                return grad

            g = grad.detach().flatten().float()  # Ensure float for histc

            # Initialize edges on first gradient (symmetric around 0)
            if self.edges is None:
                # Use robust range to avoid outlier effects
                q = torch.quantile(g.abs(), 0.98).item() or 1e-6
                r = float(q)
                self.edges = torch.linspace(-r, r, self.bins + 1, device='cpu')

            # Compute histogram
            g_cpu = g.cpu()
            hist = torch.histc(
                g_cpu.clamp(self.edges[0].item(), self.edges[-1].item()),
                bins=self.bins,
                min=self.edges[0].item(),
                max=self.edges[-1].item()
            )

            self.records.append({
                "param": pname,
                "step": self._step_counter,
                "mean": g.mean().item(),
                "std": g.std(unbiased=False).item(),
                "abs_mean": g.abs().mean().item(),
                "max": g.abs().max().item(),
                "numel": g.numel(),
                "time": time.time(),
                "hist": hist.tolist()
            })

            return grad
        return hook

    def enable(self):
        """Enable gradient recording."""
        self._enabled = True

    def disable(self):
        """Disable gradient recording (hooks still registered but do nothing)."""
        self._enabled = False

    def reset(self):
        """Clear all recorded data and reset edges."""
        self.records = []
        self.edges = None
        self._step_counter = 0

    def get_summary(self):
        """Get summary statistics for each parameter."""
        if not self.records:
            return {}

        summary = defaultdict(lambda: {
            'count': 0,
            'mean_avg': 0.0,
            'std_avg': 0.0,
            'max_avg': 0.0,
            'max_max': 0.0,
        })

        for r in self.records:
            s = summary[r['param']]
            s['count'] += 1
            s['mean_avg'] += r['mean']
            s['std_avg'] += r['std']
            s['max_avg'] += r['max']
            s['max_max'] = max(s['max_max'], r['max'])

        # Compute averages
        for param, s in summary.items():
            if s['count'] > 0:
                s['mean_avg'] /= s['count']
                s['std_avg'] /= s['count']
                s['max_avg'] /= s['count']

        return dict(summary)

    def export_csv(self, path, append=False):
        """Export recorded data to CSV file.

        Args:
            path: Output CSV file path
            append: If True, append to existing file (no header if file exists)
        """
        if not self.records:
            return

        import os
        file_exists = os.path.exists(path)
        mode = "a" if append else "w"
        write_header = not (append and file_exists)

        with open(path, mode, newline="") as f:
            w = csv.writer(f)

            # Header (only if new file or not appending)
            if write_header:
                header = ["time", "step", "param", "numel", "mean", "std", "abs_mean", "max"]
                header += [f"b{i}" for i in range(len(self.records[0]["hist"]))]
                w.writerow(header)

            # Data rows
            for r in self.records:
                row = [
                    r["time"],
                    r["step"],
                    r["param"],
                    r["numel"],
                    r["mean"],
                    r["std"],
                    r["abs_mean"],
                    r["max"]
                ]
                row += r["hist"]
                w.writerow(row)

        num_records = len(self.records)
        self.records = []  # Clear after export to avoid duplicates
        print(f"📊 GradHistogram: Appended {num_records} records to {path}")

    def export_tensorboard(self, writer, global_step, prefix="grad_hist"):
        """Export histograms to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter
            global_step: Current training step
            prefix: Tag prefix for TensorBoard
        """
        if not self.records or self.edges is None:
            return

        # Get most recent record for each parameter
        latest_by_param = {}
        for r in self.records:
            latest_by_param[r['param']] = r

        for param, r in latest_by_param.items():
            # Convert histogram to TensorBoard format
            hist = torch.tensor(r['hist'])

            # Log histogram
            tag = f"{prefix}/{param}"
            writer.add_histogram(tag, hist, global_step, bins=self.edges.tolist())

            # Log scalar statistics
            writer.add_scalar(f"{prefix}_stats/{param}/mean", r['mean'], global_step)
            writer.add_scalar(f"{prefix}_stats/{param}/std", r['std'], global_step)
            writer.add_scalar(f"{prefix}_stats/{param}/max", r['max'], global_step)

    def close(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []
        print(f"📊 GradHistogram: Closed and removed all hooks")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_cnn_grad_monitor(model, log_every_n_steps=100):
    """Factory function to create a gradient monitor for CNN models.

    Monitors:
    - All conv layer weights (conv1, conv2, conv3, conv4, conv5)
    - BottleneckBlock internal convolutions
    - FC layer weights

    Args:
        model: CNN model instance
        log_every_n_steps: Record gradients every N backward passes

    Returns:
        GradHistogram instance
    """
    def cnn_filter(module, name):
        # Match conv stages and their internal layers
        if any(k in name for k in ["conv1", "conv2", "conv3", "conv4", "conv5"]):
            return True
        # Match FC layer
        if "fc" in name:
            return True
        # Match BatchNorm layers (gamma/beta can reveal training issues)
        if "bn" in name:
            return True
        return False

    return GradHistogram(
        model,
        modules_filter=cnn_filter,
        bins=41,
        track_layers=("weight",),
        log_every_n_steps=log_every_n_steps,
    )
