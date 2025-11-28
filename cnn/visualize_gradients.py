#!/usr/bin/env python3
"""Visualize gradient histograms during or after training.

Usage:
    python visualize_gradients.py results/cnn_results_prelu/grad_histogram.csv
    python visualize_gradients.py results/cnn_results_prelu/grad_histogram.csv --watch
    python visualize_gradients.py results/cnn_results_prelu/grad_histogram.csv --save
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path):
    """Load gradient histogram CSV."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def get_hist_cols(df):
    """Get histogram bin columns."""
    return [c for c in df.columns if c.startswith('b')]


def plot_summary(df, fig=None):
    """Plot summary dashboard."""
    if fig is None:
        fig = plt.figure(figsize=(16, 10))
    else:
        fig.clf()

    # Get unique steps and params
    steps = sorted(df['step'].unique())
    params = df['param'].unique()
    latest_step = steps[-1]

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # --- Plot 1: Gradient stats over time ---
    ax1 = fig.add_subplot(gs[0, 0])
    step_stats = df.groupby('step').agg({
        'std': 'mean',
        'max': 'max',
        'abs_mean': 'mean'
    }).reset_index()

    ax1.semilogy(step_stats['step'], step_stats['std'], label='std (avg)', linewidth=2)
    ax1.semilogy(step_stats['step'], step_stats['max'], label='max', alpha=0.7)
    ax1.semilogy(step_stats['step'], step_stats['abs_mean'], label='|mean| (avg)', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Gradient magnitude (log)')
    ax1.set_title('Gradient Magnitude Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Layer comparison (latest step) ---
    ax2 = fig.add_subplot(gs[0, 1])
    latest = df[df['step'] == latest_step].copy()
    latest = latest.sort_values('std', ascending=True)

    # Truncate long names for display
    display_names = [p.replace('.weight', '').replace('conv', 'c')[-30:] for p in latest['param']]

    y_pos = range(len(latest))
    ax2.barh(y_pos, latest['std'].values, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_names, fontsize=6)
    ax2.set_xlabel('Gradient std')
    ax2.set_title(f'Gradient Spread by Layer (step {latest_step})')

    # --- Plot 3: Heatmap of gradient std over time ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Pivot to get layers x steps
    pivot = df.pivot_table(index='param', columns='step', values='std', aggfunc='first')

    # Normalize per row for visibility
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0).fillna(0)

    im = ax3.imshow(pivot_norm.values, aspect='auto', cmap='viridis')
    ax3.set_xlabel('Step index')
    ax3.set_ylabel('Layer')
    ax3.set_title('Gradient Std Evolution (normalized per-layer)')

    # Only show a few y-tick labels
    n_layers = len(pivot.index)
    if n_layers > 20:
        tick_idx = np.linspace(0, n_layers-1, 10, dtype=int)
        ax3.set_yticks(tick_idx)
        ax3.set_yticklabels([pivot.index[i][-25:] for i in tick_idx], fontsize=6)
    else:
        ax3.set_yticks(range(n_layers))
        ax3.set_yticklabels([p[-25:] for p in pivot.index], fontsize=6)

    # --- Plot 4: Top changing layers ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Find layers with most change in gradient std
    if len(steps) > 1:
        layer_change = df.groupby('param')['std'].agg(['first', 'last', 'std'])
        layer_change['change'] = abs(layer_change['last'] - layer_change['first'])
        top_layers = layer_change.nlargest(8, 'change').index.tolist()

        for param in top_layers:
            layer_df = df[df['param'] == param].sort_values('step')
            label = param.replace('.weight', '')[-25:]
            ax4.plot(layer_df['step'], layer_df['std'], label=label, alpha=0.8)

        ax4.set_xlabel('Step')
        ax4.set_ylabel('Gradient std')
        ax4.set_title('Top 8 Most Changing Layers')
        ax4.legend(fontsize=7, loc='upper right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Need more steps\nfor evolution plot',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Layer Evolution (need more data)')

    # Add overall title
    fig.suptitle(f'Gradient Histogram Summary | Steps: {steps[0]}-{latest_step} | Layers: {len(params)}',
                 fontsize=12, fontweight='bold')

    return fig


def print_summary(df):
    """Print text summary."""
    steps = sorted(df['step'].unique())
    latest = df[df['step'] == steps[-1]]

    print(f"\n{'='*60}")
    print(f"GRADIENT SUMMARY | Step {steps[-1]} | {len(latest)} layers")
    print(f"{'='*60}")

    # Top by std
    print("\nTop 5 by gradient std:")
    top = latest.nlargest(5, 'std')[['param', 'std', 'max']]
    for _, row in top.iterrows():
        print(f"  {row['param'][-40:]}: std={row['std']:.2e}, max={row['max']:.2e}")

    # Bottom by std (potential vanishing)
    print("\nBottom 5 by gradient std:")
    bottom = latest.nsmallest(5, 'std')[['param', 'std', 'max']]
    for _, row in bottom.iterrows():
        print(f"  {row['param'][-40:]}: std={row['std']:.2e}, max={row['max']:.2e}")

    # Warnings
    tiny = latest[latest['std'] < 1e-7]
    huge = latest[latest['max'] > 10]

    if len(tiny) > 0:
        print(f"\n⚠️  {len(tiny)} layers with tiny gradients (std < 1e-7)")
    if len(huge) > 0:
        print(f"⚠️  {len(huge)} layers with large gradients (max > 10)")
    if len(tiny) == 0 and len(huge) == 0:
        print("\n✅ Gradients look healthy")

    print(f"{'='*60}\n")


def watch_mode(path, interval=10):
    """Watch file and update plot periodically."""
    print(f"Watching {path} (refresh every {interval}s, Ctrl+C to stop)")

    plt.ion()
    fig = plt.figure(figsize=(16, 10))

    last_size = 0

    try:
        while True:
            # Check if file changed
            if os.path.exists(path):
                current_size = os.path.getsize(path)
                if current_size != last_size:
                    last_size = current_size

                    df = load_data(path)
                    if df is not None and len(df) > 0:
                        plot_summary(df, fig)
                        print_summary(df)
                        fig.canvas.draw()
                        fig.canvas.flush_events()
            else:
                print(f"Waiting for {path}...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching")
    finally:
        plt.ioff()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize gradient histograms")
    parser.add_argument("path", help="Path to grad_histogram.csv")
    parser.add_argument("--watch", action="store_true", help="Watch mode: auto-refresh during training")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval for watch mode (seconds)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.path, args.interval)
    else:
        df = load_data(args.path)
        if df is None:
            print(f"No data found at {args.path}")
            sys.exit(1)

        print_summary(df)
        fig = plot_summary(df)

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
