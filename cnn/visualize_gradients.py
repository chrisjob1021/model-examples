#!/usr/bin/env python3
"""Diagnostic gradient visualization for ResNet/CNN training.

Detects:
- Collapsing gradients in later residual blocks
- Starved upper layers after pooling/downsampling
- Heavy tails indicating instability

Usage:
    python visualize_gradients.py results/cnn_results_prelu/grad_histogram.csv
    python visualize_gradients.py results/cnn_results_prelu/grad_histogram.csv --watch
"""

import argparse
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


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


def parse_layer_info(param_name):
    """Extract stage and block info from parameter name."""
    # Examples:
    #   conv1.0.conv.weight -> stage=1, block=0, type=conv
    #   _orig_mod.conv1.0.bn.weight -> stage=1, block=0, type=bn
    #   conv3.2.bn1.weight -> stage=3, block=2, type=bn
    #   conv5.0.shortcut.0.weight -> stage=5, block=0, type=shortcut
    #   fc.weight -> stage=fc, block=0, type=fc

    # Remove common prefixes (e.g., _orig_mod., module., model.)
    clean_name = param_name
    for prefix in ['_orig_mod.', 'module.', 'model.']:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break

    if clean_name.startswith('fc') or param_name.endswith('.fc.weight'):
        return {'stage': 'fc', 'block': 0, 'type': 'fc', 'full': param_name}

    # Match conv{N}.{block}... (with optional prefix already removed)
    # Also handle cases like conv{N}.{block}.{layer}.{sub}.weight
    match = re.match(r'conv(\d+)\.(\d+)\.(.+?)\.weight', clean_name)
    if match:
        stage = int(match.group(1))
        block = int(match.group(2))
        layer_type = match.group(3)

        # Categorize layer type
        if 'shortcut' in layer_type:
            ltype = 'shortcut'
        elif 'bn' in layer_type:
            ltype = 'bn'
        elif 'conv' in layer_type:
            ltype = 'conv'
        elif 'act' in layer_type or 'prelu' in layer_type.lower():
            ltype = 'act'
        else:
            ltype = layer_type

        return {'stage': stage, 'block': block, 'type': ltype, 'full': param_name}

    # Fallback
    return {'stage': 'other', 'block': 0, 'type': 'other', 'full': param_name}


def compute_histogram_stats(row, hist_cols):
    """Compute histogram shape statistics."""
    hist = np.array([row[c] for c in hist_cols])
    total = hist.sum()
    if total == 0:
        return {'kurtosis': 0, 'tail_ratio': 0, 'center_mass': 0}

    hist_norm = hist / total
    n_bins = len(hist)
    center = n_bins // 2

    # Kurtosis proxy: ratio of tails to center
    tail_bins = 5  # Number of bins on each tail
    center_bins = 10  # Number of bins in center

    tail_mass = hist_norm[:tail_bins].sum() + hist_norm[-tail_bins:].sum()
    center_start = center - center_bins // 2
    center_end = center + center_bins // 2
    center_mass = hist_norm[center_start:center_end].sum()

    tail_ratio = tail_mass / (center_mass + 1e-10)

    return {
        'tail_ratio': tail_ratio,
        'center_mass': center_mass,
    }


def diagnose_issues(df):
    """Diagnose gradient issues and return warnings."""
    warnings = []
    hist_cols = get_hist_cols(df)

    # Get latest step data
    latest_step = df['step'].max()
    latest = df[df['step'] == latest_step].copy()

    # Parse layer info
    latest['layer_info'] = latest['param'].apply(parse_layer_info)
    latest['stage'] = latest['layer_info'].apply(lambda x: x['stage'])
    latest['block'] = latest['layer_info'].apply(lambda x: x['block'])
    latest['ltype'] = latest['layer_info'].apply(lambda x: x['type'])

    # 1. Check for collapsing gradients in later stages
    conv_layers = latest[latest['stage'].apply(lambda x: isinstance(x, int))]
    if len(conv_layers) > 0:
        stage_stats = conv_layers.groupby('stage')['std'].agg(['mean', 'min', 'max'])

        # Check if later stages have much smaller gradients
        if len(stage_stats) >= 3:
            early_std = stage_stats.loc[stage_stats.index <= 2, 'mean'].mean()
            late_std = stage_stats.loc[stage_stats.index >= 4, 'mean'].mean()

            if late_std < early_std * 0.1:
                warnings.append({
                    'type': 'COLLAPSING_GRADIENTS',
                    'severity': 'HIGH',
                    'msg': f'Later stages (conv4-5) have 10x smaller gradients than early stages',
                    'detail': f'Early avg std: {early_std:.2e}, Late avg std: {late_std:.2e}',
                    'fix': 'Consider: higher LR warmup, LayerScale, or adjust norm placement'
                })
            elif late_std < early_std * 0.3:
                warnings.append({
                    'type': 'WEAK_LATE_GRADIENTS',
                    'severity': 'MEDIUM',
                    'msg': f'Later stages have notably weaker gradients',
                    'detail': f'Early avg std: {early_std:.2e}, Late avg std: {late_std:.2e}',
                    'fix': 'Monitor for convergence issues; may need gradient scaling'
                })

    # 2. Check for starved layers after downsampling
    shortcut_layers = latest[latest['ltype'] == 'shortcut']
    if len(shortcut_layers) > 0:
        shortcut_std = shortcut_layers['std'].mean()
        conv_std = latest[latest['ltype'] == 'conv']['std'].mean()

        if shortcut_std < conv_std * 0.05:
            warnings.append({
                'type': 'STARVED_SHORTCUTS',
                'severity': 'MEDIUM',
                'msg': 'Shortcut/downsample paths have very weak gradients',
                'detail': f'Shortcut std: {shortcut_std:.2e}, Conv std: {conv_std:.2e}',
                'fix': 'Try: blur-pool, reduce weight decay on downsample, or widen bottleneck'
            })

    # 3. Check for heavy tails (instability)
    for _, row in latest.iterrows():
        hist_stats = compute_histogram_stats(row, hist_cols)
        if hist_stats['tail_ratio'] > 0.5:  # More mass in tails than expected
            stage_info = row['layer_info']
            if stage_info['stage'] in [1, 2]:  # Early stages - some tails OK
                continue
            warnings.append({
                'type': 'HEAVY_TAILS',
                'severity': 'LOW',
                'msg': f"Heavy gradient tails in {row['param'][-40:]}",
                'detail': f"Tail ratio: {hist_stats['tail_ratio']:.2f}",
                'fix': 'Check for gradient spikes; may need clipping or LR reduction'
            })

    # 4. Check for vanishing gradients
    vanishing = latest[latest['std'] < 1e-7]
    if len(vanishing) > 0:
        warnings.append({
            'type': 'VANISHING_GRADIENTS',
            'severity': 'HIGH',
            'msg': f'{len(vanishing)} layers with near-zero gradients',
            'detail': ', '.join(vanishing['param'].values[:5]),
            'fix': 'Check for dead ReLUs, increase LR, or use PReLU/LeakyReLU'
        })

    # 5. Check gradient evolution (are they stabilizing?)
    if len(df['step'].unique()) > 5:
        early_steps = df[df['step'] <= df['step'].quantile(0.2)]
        late_steps = df[df['step'] >= df['step'].quantile(0.8)]

        early_std_var = early_steps.groupby('param')['std'].std().mean()
        late_std_var = late_steps.groupby('param')['std'].std().mean()

        if late_std_var > early_std_var * 2:
            warnings.append({
                'type': 'UNSTABLE_GRADIENTS',
                'severity': 'MEDIUM',
                'msg': 'Gradient variance increasing over training',
                'detail': f'Early variance: {early_std_var:.2e}, Late variance: {late_std_var:.2e}',
                'fix': 'Consider LR decay, gradient clipping, or longer warmup'
            })

    return warnings


def plot_diagnostic(df, fig=None):
    """Plot diagnostic dashboard for residual networks."""
    if fig is None:
        fig = plt.figure(figsize=(18, 12))
    else:
        fig.clf()

    hist_cols = get_hist_cols(df)
    steps = sorted(df['step'].unique())
    latest_step = steps[-1]

    # Parse layer info for all rows
    df = df.copy()
    df['layer_info'] = df['param'].apply(parse_layer_info)
    df['stage'] = df['layer_info'].apply(lambda x: x['stage'])
    df['block'] = df['layer_info'].apply(lambda x: x['block'])
    df['ltype'] = df['layer_info'].apply(lambda x: x['type'])

    latest = df[df['step'] == latest_step]

    # Layout: 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # --- Plot 1: Per-stage gradient std over time ---
    ax1 = fig.add_subplot(gs[0, 0])

    conv_df = df[df['stage'].apply(lambda x: isinstance(x, int))]
    stage_time = conv_df.groupby(['step', 'stage'])['std'].mean().reset_index()
    
    # Clean data: remove NaN/inf and ensure numeric types
    stage_time = stage_time.copy()
    stage_time['std'] = pd.to_numeric(stage_time['std'], errors='coerce')
    stage_time = stage_time.dropna(subset=['std', 'step'])
    stage_time = stage_time[np.isfinite(stage_time['std'])]

    has_data = False
    for stage in sorted(stage_time['stage'].unique()):
        stage_data = stage_time[stage_time['stage'] == stage]
        if len(stage_data) > 0 and stage_data['std'].notna().any():
            # Filter out any remaining invalid values
            valid = np.isfinite(stage_data['std']) & (stage_data['std'] > 0)
            if valid.any():
                ax1.semilogy(stage_data.loc[valid, 'step'], stage_data.loc[valid, 'std'],
                             label=f'conv{stage}', linewidth=2, alpha=0.8)
                has_data = True

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Gradient std (log)')
    ax1.set_title('Per-Stage Gradient Evolution\n(should stabilize, not collapse)')
    if has_data:
        ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add reference lines
    ax1.axhline(y=1e-4, color='r', linestyle='--', alpha=0.3, label='warning threshold')

    # --- Plot 2: Stage-by-stage comparison (latest) ---
    ax2 = fig.add_subplot(gs[0, 1])

    conv_latest = latest[latest['stage'].apply(lambda x: isinstance(x, int))]
    stage_stats = conv_latest.groupby('stage')['std'].agg(['mean', 'min', 'max']).reset_index()
    
    # Clean data: ensure numeric and remove invalid values
    stage_stats = stage_stats.copy()
    for col in ['mean', 'min', 'max']:
        stage_stats[col] = pd.to_numeric(stage_stats[col], errors='coerce')
    stage_stats = stage_stats.dropna(subset=['mean', 'min', 'max'])
    stage_stats = stage_stats[np.isfinite(stage_stats['mean']) & 
                               np.isfinite(stage_stats['min']) & 
                               np.isfinite(stage_stats['max'])]

    if len(stage_stats) > 0:
        x = stage_stats['stage'].values
        means = stage_stats['mean'].values
        mins = stage_stats['min'].values
        maxs = stage_stats['max'].values
        
        ax2.bar(x - 0.2, means, 0.4, label='mean', alpha=0.8)
        ax2.errorbar(x - 0.2, means,
                     yerr=[means - mins, maxs - means],
                     fmt='none', color='black', capsize=3)

        ax2.set_xlabel('Conv Stage')
        ax2.set_ylabel('Gradient std')
        ax2.set_title(f'Stage Comparison (step {latest_step})\n(later stages should not be much smaller)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'conv{int(s)}' for s in x])

        # Highlight if later stages are weak
        if len(stage_stats) >= 4:
            early_stats = stage_stats[stage_stats['stage'] <= 2]
            late_stats = stage_stats[stage_stats['stage'] >= 4]
            if len(early_stats) > 0 and len(late_stats) > 0:
                early_mean = early_stats['mean'].mean()
                late_mean = late_stats['mean'].mean()
                if np.isfinite(early_mean) and np.isfinite(late_mean) and late_mean < early_mean * 0.3:
                    ax2.axhline(y=early_mean * 0.3, color='r', linestyle='--', alpha=0.5)
                    ax2.text(0.95, 0.95, '[WARNING] Late stages weak!', transform=ax2.transAxes,
                            ha='right', va='top', color='red', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No stage data available', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_xlabel('Conv Stage')
        ax2.set_ylabel('Gradient std')
        ax2.set_title(f'Stage Comparison (step {latest_step})')

    # --- Plot 3: Block-level within each stage ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Show gradient std for each block within stages
    conv_latest_sorted = conv_latest.sort_values(['stage', 'block'])

    # Group by stage and plot blocks
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    block_data = []
    for stage in sorted(conv_latest['stage'].unique()):
        if not isinstance(stage, int):
            continue
        stage_blocks = conv_latest[conv_latest['stage'] == stage].groupby('block')['std'].mean()
        for block, std_val in stage_blocks.items():
            block_data.append({'stage': stage, 'block': block, 'std': std_val})

    block_df = pd.DataFrame(block_data)
    if len(block_df) > 0 and 'stage' in block_df.columns:
        # Clean data: ensure numeric and remove invalid values
        block_df = block_df.copy()
        block_df['std'] = pd.to_numeric(block_df['std'], errors='coerce')
        block_df = block_df.dropna(subset=['std', 'block'])
        block_df = block_df[np.isfinite(block_df['std'])]
        
        for stage in sorted(block_df['stage'].unique()):
            stage_data = block_df[block_df['stage'] == stage]
            if len(stage_data) > 0:
                ax3.plot(stage_data['block'], stage_data['std'], 'o-',
                        label=f'conv{stage}', linewidth=2, markersize=8)
    else:
        ax3.text(0.5, 0.5, 'No block data available', 
                transform=ax3.transAxes, ha='center', va='center')

    ax3.set_xlabel('Block index within stage')
    ax3.set_ylabel('Gradient std')
    ax3.set_title('Per-Block Gradients\n(should be stable within stage, not declining)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Layer type comparison ---
    ax4 = fig.add_subplot(gs[1, 1])

    type_stats = latest.groupby('ltype')['std'].agg(['mean', 'count']).reset_index()
    
    # Clean data: ensure numeric and remove invalid values
    type_stats = type_stats.copy()
    type_stats['mean'] = pd.to_numeric(type_stats['mean'], errors='coerce')
    type_stats = type_stats.dropna(subset=['mean'])
    type_stats = type_stats[np.isfinite(type_stats['mean'])]
    type_stats = type_stats.sort_values('mean', ascending=True)

    if len(type_stats) > 0:
        colors = ['red' if t == 'shortcut' else 'steelblue' for t in type_stats['ltype']]
        bars = ax4.barh(type_stats['ltype'], type_stats['mean'], color=colors, alpha=0.8)

        ax4.set_xlabel('Gradient std')
        ax4.set_title('By Layer Type\n(shortcuts should not be much weaker than convs)')

        # Add count labels
        for i, (_, row) in enumerate(type_stats.iterrows()):
            if np.isfinite(row['mean']):
                ax4.text(row['mean'], i, f" n={int(row['count'])}", va='center', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No layer type data available', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_xlabel('Gradient std')
        ax4.set_title('By Layer Type')

    # --- Plot 5: Histogram shape (tails vs center) ---
    ax5 = fig.add_subplot(gs[2, 0])

    # Compute tail ratios for all layers at latest step
    tail_data = []
    for _, row in latest.iterrows():
        hist_stats = compute_histogram_stats(row, hist_cols)
        info = row['layer_info']
        tail_data.append({
            'param': row['param'],
            'stage': info['stage'],
            'tail_ratio': hist_stats['tail_ratio'],
            'std': row['std']
        })

    tail_df = pd.DataFrame(tail_data)
    tail_df = tail_df[tail_df['stage'].apply(lambda x: isinstance(x, int))]
    
    # Clean data: ensure numeric and remove invalid values
    tail_df = tail_df.copy()
    tail_df['std'] = pd.to_numeric(tail_df['std'], errors='coerce')
    tail_df['tail_ratio'] = pd.to_numeric(tail_df['tail_ratio'], errors='coerce')
    tail_df = tail_df.dropna(subset=['std', 'tail_ratio'])
    tail_df = tail_df[np.isfinite(tail_df['std']) & np.isfinite(tail_df['tail_ratio'])]

    # Scatter: std vs tail_ratio, colored by stage
    for stage in sorted(tail_df['stage'].unique()):
        stage_data = tail_df[tail_df['stage'] == stage]
        if len(stage_data) > 0:
            ax5.scatter(stage_data['std'], stage_data['tail_ratio'],
                       label=f'conv{stage}', alpha=0.7, s=50)

    ax5.set_xlabel('Gradient std')
    ax5.set_ylabel('Tail ratio (higher = heavier tails)')
    ax5.set_title('Histogram Shape\n(heavy tails in later stages = instability)')
    ax5.axhline(y=0.3, color='r', linestyle='--', alpha=0.3)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- Plot 6: Diagnostics summary ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    warnings = diagnose_issues(df)

    text = "DIAGNOSTIC SUMMARY\n" + "=" * 40 + "\n\n"

    if not warnings:
        text += "[OK] No major issues detected\n\n"
        text += "Gradients appear healthy:\n"
        text += "• Per-stage std is stable\n"
        text += "• No collapsing in later blocks\n"
        text += "• Shortcut paths are active\n"
    else:
        for w in warnings:
            severity_color = {'HIGH': '[HIGH]', 'MEDIUM': '[MED]', 'LOW': '[LOW]'}
            text += f"{severity_color.get(w['severity'], '[?]')} {w['type']}\n"
            text += f"   {w['msg']}\n"
            text += f"   {w['detail']}\n"
            text += f"   → {w['fix']}\n\n"

    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    n_layers = latest['param'].nunique()
    fig.suptitle(f'Gradient Diagnostics | Steps: {steps[0]}-{latest_step} | Layers: {n_layers}',
                 fontsize=14, fontweight='bold')

    return fig, warnings


def print_diagnostics(df):
    """Print diagnostic summary."""
    warnings = diagnose_issues(df)

    print(f"\n{'='*60}")
    print("GRADIENT DIAGNOSTICS")
    print(f"{'='*60}")

    if not warnings:
        print("\n[OK] No major issues detected - gradients look healthy\n")
    else:
        for w in warnings:
            severity_icon = {'HIGH': '[HIGH]', 'MEDIUM': '[MED]', 'LOW': '[LOW]'}
            print(f"\n{severity_icon.get(w['severity'], '[?]')} [{w['severity']}] {w['type']}")
            print(f"   {w['msg']}")
            print(f"   {w['detail']}")
            print(f"   Fix: {w['fix']}")

    print(f"\n{'='*60}\n")

    return warnings


def watch_mode(path, interval=10):
    """Watch file and update plot periodically."""
    print(f"Watching {path} (refresh every {interval}s, Ctrl+C to stop)")

    plt.ion()
    fig = plt.figure(figsize=(18, 12))

    last_size = 0

    try:
        while True:
            if os.path.exists(path):
                current_size = os.path.getsize(path)
                if current_size != last_size:
                    last_size = current_size

                    df = load_data(path)
                    if df is not None and len(df) > 0:
                        plot_diagnostic(df, fig)
                        print_diagnostics(df)
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
    parser = argparse.ArgumentParser(description="Diagnostic gradient visualization")
    parser.add_argument("path", nargs='?', default="results/cnn_results_prelu/grad_histogram.csv",
                       help="Path to grad_histogram.csv (default: results/cnn_results_prelu/grad_histogram.csv)")
    parser.add_argument("--watch", action="store_true", help="Watch mode: auto-refresh")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file (overrides default)")
    parser.add_argument("--no-save", action="store_true", help="Don't save plot, just display it")
    args = parser.parse_args()

    if args.watch:
        watch_mode(args.path, args.interval)
    else:
        df = load_data(args.path)
        if df is None:
            print(f"No data found at {args.path}")
            sys.exit(1)

        warnings = print_diagnostics(df)
        fig, _ = plot_diagnostic(df)

        # Determine save path
        if not args.no_save:
            if args.save:
                save_path = args.save
            else:
                # Generate default save path from input file
                base_dir = os.path.dirname(os.path.abspath(args.path))
                base_name = os.path.splitext(os.path.basename(args.path))[0]
                # Remove common suffixes for cleaner name
                base_name = base_name.replace('grad_histogram', 'diagnostics').replace('_histogram', '')
                save_path = os.path.join(base_dir, f"{base_name}.png")
            
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(os.path.abspath(save_path))
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Created directory: {save_dir}")
            
            try:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {save_path}")
            except Exception as e:
                print(f"Error saving to {save_path}: {e}", file=sys.stderr)
                sys.exit(1)
            
            # Close figure after saving
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
