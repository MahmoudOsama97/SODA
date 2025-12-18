"""
Generate the Target Drift Analysis figure (Figure 5).
Compares SODA reservoir stability vs ERM batch instability.

Usage:
    python plot_target_drift.py --erm_file ./drift_results/drift_erm_officehome.json \
                                 --soda_file ./drift_results/drift_soda_officehome.json \
                                 --output ./fig/target_drift.pdf
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import argparse


def load_drift_data(filepath: str) -> dict:
    """Load drift data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def smooth_curve(values: list, sigma: float = 5.0) -> np.ndarray:
    """Apply Gaussian smoothing for visualization."""
    return gaussian_filter1d(np.array(values), sigma=sigma)


def plot_target_drift(
    erm_file: str,
    soda_file: str,
    output_path: str = "./fig/target_drift.pdf",
    smooth_sigma: float = 5.0,
    figsize: tuple = (10, 6)
):
    """
    Generate the Target Drift comparison figure.
    
    Args:
        erm_file: Path to ERM drift JSON
        soda_file: Path to SODA drift JSON  
        output_path: Where to save the figure
        smooth_sigma: Smoothing parameter for visualization
        figsize: Figure size (width, height)
    """
    # Load data
    print(f"Loading ERM data from: {erm_file}")
    erm_data = load_drift_data(erm_file)
    print(f"Loading SODA data from: {soda_file}")
    soda_data = load_drift_data(soda_file)
    
    # Extract values
    erm_iters = np.array(erm_data['iterations'])
    erm_drift = np.array(erm_data['drift_values'])
    soda_iters = np.array(soda_data['iterations'])
    soda_drift = np.array(soda_data['drift_values'])
    
    print(f"ERM: {len(erm_iters)} points, drift range: [{erm_drift.min():.4f}, {erm_drift.max():.4f}]")
    print(f"SODA: {len(soda_iters)} points, drift range: [{soda_drift.min():.4f}, {soda_drift.max():.4f}]")
    
    # Skip first few iterations (warmup/initialization)
    warmup = min(20, len(erm_iters) // 10)
    erm_iters = erm_iters[warmup:]
    erm_drift = erm_drift[warmup:]
    soda_iters = soda_iters[warmup:]
    soda_drift = soda_drift[warmup:]
    
    # Replace any NaN or Inf values
    erm_drift = np.nan_to_num(erm_drift, nan=0.0, posinf=0.0, neginf=0.0)
    soda_drift = np.nan_to_num(soda_drift, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply smoothing for cleaner visualization
    erm_smooth = smooth_curve(erm_drift, smooth_sigma)
    soda_smooth = smooth_curve(soda_drift, smooth_sigma)
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'font.family': 'serif',
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data with low alpha (shows variance)
    ax.fill_between(erm_iters, 
                    np.maximum(erm_drift - 0.02, 0), 
                    erm_drift + 0.02, 
                    color='red', alpha=0.15, label='_nolegend_')
    ax.fill_between(soda_iters, 
                    np.maximum(soda_drift - 0.01, 0), 
                    soda_drift + 0.01, 
                    color='blue', alpha=0.15, label='_nolegend_')
    
    # Plot smoothed curves
    ax.plot(erm_iters, erm_smooth, color='#d62728', linestyle='--', linewidth=2.5,
            label='Standard Batch-Derived Pivot (ERM)', marker='', markersize=0)
    ax.plot(soda_iters, soda_smooth, color='#1f77b4', linestyle='-', linewidth=2.5,
            label='Dynamic Feature Reservoir (SODA)', marker='', markersize=0)
    
    # Labels and formatting
    ax.set_xlabel('Training Iterations', fontsize=14, fontweight='medium')
    ax.set_ylabel(r'Target Drift: $W_2(P_{pivot}^{(t)}, P_{pivot}^{(t+1)})$', fontsize=14, fontweight='medium')
    ax.set_title('Quantitative Stability Analysis', fontsize=16, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, 
              shadow=True, borderpad=1)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Axis limits
    ax.set_xlim([erm_iters[0], erm_iters[-1]])
    y_max = max(erm_smooth.max(), soda_smooth.max()) * 1.2
    ax.set_ylim(bottom=0, top=y_max)
    
    # Add annotations
    # Find good positions for annotations
    mid_idx = len(soda_iters) * 3 // 4
    erm_mid_idx = len(erm_iters) * 3 // 4
    
    # Annotation for SODA (stable regime)
    if soda_smooth[mid_idx] < y_max * 0.5:
        ax.annotate('Stable regime\n$(K \\gg B)$', 
                    xy=(soda_iters[mid_idx], soda_smooth[mid_idx]),
                    xytext=(soda_iters[mid_idx] - len(soda_iters)*0.15, soda_smooth[mid_idx] + y_max * 0.15),
                    fontsize=10, color='#1f77b4', fontweight='medium',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', alpha=0.8, lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.8))
    
    # Annotation for ERM (persistent instability)
    if erm_smooth[erm_mid_idx] > y_max * 0.3:
        ax.annotate('Persistent\ninstability\n$(B \\ll d\')$', 
                    xy=(erm_iters[erm_mid_idx], erm_smooth[erm_mid_idx]),
                    xytext=(erm_iters[erm_mid_idx] - len(erm_iters)*0.2, erm_smooth[erm_mid_idx] - y_max * 0.15),
                    fontsize=10, color='#d62728', fontweight='medium',
                    arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.8, lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728', alpha=0.8))
    
    # Add vertical line at warmup end (optional)
    # ax.axvline(x=warmup*10, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
    
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PDF figure to: {output_path}")
    
    # Also save PNG version
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved PNG figure to: {png_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"ERM Mean Drift:  {erm_drift.mean():.4f} ± {erm_drift.std():.4f}")
    print(f"SODA Mean Drift: {soda_drift.mean():.4f} ± {soda_drift.std():.4f}")
    print(f"Stability Ratio (ERM/SODA): {erm_drift.mean() / max(soda_drift.mean(), 1e-6):.2f}x")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Plot Target Drift comparison figure")
    parser.add_argument('--erm_file', type=str, 
                        default='./drift_results/drift_erm_officehome.json',
                        help='Path to ERM drift JSON file')
    parser.add_argument('--soda_file', type=str,
                        default='./drift_results/drift_soda_officehome.json',
                        help='Path to SODA drift JSON file')
    parser.add_argument('--output', type=str, default='./fig/target_drift.pdf',
                        help='Output path for the figure')
    parser.add_argument('--smooth', type=float, default=5.0,
                        help='Gaussian smoothing sigma')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', '..')
    
    if not os.path.isabs(args.erm_file):
        args.erm_file = os.path.join(base_dir, args.erm_file)
    if not os.path.isabs(args.soda_file):
        args.soda_file = os.path.join(base_dir, args.soda_file)
    if not os.path.isabs(args.output):
        args.output = os.path.join(base_dir, args.output)
    
    # Check files exist
    if not os.path.exists(args.erm_file):
        print(f"Error: ERM file not found: {args.erm_file}")
        print("Run train_drift_comparison.py first to generate the data.")
        sys.exit(1)
    if not os.path.exists(args.soda_file):
        print(f"Error: SODA file not found: {args.soda_file}")
        print("Run train_drift_comparison.py first to generate the data.")
        sys.exit(1)
    
    plot_target_drift(
        erm_file=args.erm_file,
        soda_file=args.soda_file,
        output_path=args.output,
        smooth_sigma=args.smooth
    )


if __name__ == "__main__":
    main()
