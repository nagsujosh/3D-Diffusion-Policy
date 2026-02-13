"""
Utility script to visualize and analyze component-wise action losses from training logs.

Usage:
    python scripts/analyze_action_losses.py --wandb_run <run_id>
    python scripts/analyze_action_losses.py --log_file <path_to_log.csv>
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_action_losses(data, output_dir="loss_plots"):
    """
    Plot component-wise action losses over training.
    
    Args:
        data: DataFrame with columns including step, action_pos_mse, action_rot_mse, action_gripper_mse
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Filter for action loss columns
    action_loss_cols = [col for col in data.columns if 'action' in col and 'mse' in col]
    
    if not action_loss_cols:
        print("No action loss columns found in data!")
        return
    
    print(f"Found {len(action_loss_cols)} action loss components: {action_loss_cols}")
    
    # Create figure with subplots
    n_cols = len(action_loss_cols)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: All losses together
    ax = axes[0]
    for col in action_loss_cols:
        if col in data.columns:
            ax.plot(data['step'], data[col], label=col, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Action Component Losses Over Training')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often better for loss curves
    
    # Plot 2: Normalized comparison (each loss normalized to [0,1])
    ax = axes[1]
    for col in action_loss_cols:
        if col in data.columns:
            # Normalize to [0, 1] for comparison
            values = data[col].values
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = values
            ax.plot(data['step'], normalized, label=f"{col} (normalized)", alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Loss [0-1]')
    ax.set_title('Normalized Action Component Losses (for relative comparison)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "action_losses_overview.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Create individual plots for 7-DOF standard format
    if 'action_pos_mse' in data.columns and 'action_rot_mse' in data.columns and 'action_gripper_mse' in data.columns:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Position loss
        ax = axes[0]
        ax.plot(data['step'], data['action_pos_mse'], color='blue', linewidth=2)
        ax.fill_between(data['step'], data['action_pos_mse'], alpha=0.3, color='blue')
        ax.set_ylabel('Position MSE')
        ax.set_title('Position Loss (X, Y, Z)')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Rotation loss
        ax = axes[1]
        ax.plot(data['step'], data['action_rot_mse'], color='orange', linewidth=2)
        ax.fill_between(data['step'], data['action_rot_mse'], alpha=0.3, color='orange')
        ax.set_ylabel('Rotation MSE')
        ax.set_title('Rotation Loss (Roll, Pitch, Yaw)')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Gripper loss
        ax = axes[2]
        ax.plot(data['step'], data['action_gripper_mse'], color='green', linewidth=2)
        ax.fill_between(data['step'], data['action_gripper_mse'], alpha=0.3, color='green')
        ax.set_ylabel('Gripper MSE')
        ax.set_xlabel('Training Step')
        ax.set_title('Gripper Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        output_file = output_dir / "action_losses_detailed.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("Loss Statistics Summary:")
    print("="*60)
    for col in action_loss_cols:
        if col in data.columns:
            values = data[col].values
            print(f"\n{col}:")
            print(f"  Initial: {values[0]:.6f}")
            print(f"  Final:   {values[-1]:.6f}")
            print(f"  Min:     {values.min():.6f}")
            print(f"  Max:     {values.max():.6f}")
            print(f"  Mean:    {values.mean():.6f}")
            print(f"  Std:     {values.std():.6f}")
            reduction = (values[0] - values[-1]) / values[0] * 100
            print(f"  Reduction: {reduction:.1f}%")


def load_from_wandb(run_id, project=None, entity=None):
    """Load training data from Weights & Biases."""
    try:
        import wandb
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb")
        return None
    
    api = wandb.Api()
    
    # Construct run path
    if project and entity:
        run_path = f"{entity}/{project}/{run_id}"
    elif project:
        run_path = f"{project}/{run_id}"
    else:
        run_path = run_id
    
    print(f"Fetching run: {run_path}")
    run = api.run(run_path)
    
    # Get history
    history = run.history()
    
    # Filter for relevant columns
    relevant_cols = ['_step'] + [col for col in history.columns if 'action' in col and 'mse' in col]
    relevant_cols += ['train_loss', 'bc_loss']
    
    data = history[relevant_cols].copy()
    data.rename(columns={'_step': 'step'}, inplace=True)
    
    # Drop NaN rows
    data = data.dropna()
    
    return data


def load_from_csv(csv_path):
    """Load training data from CSV file."""
    data = pd.read_csv(csv_path)
    
    if 'step' not in data.columns and 'global_step' in data.columns:
        data.rename(columns={'global_step': 'step'}, inplace=True)
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Analyze component-wise action losses")
    parser.add_argument('--wandb_run', type=str, help='WandB run ID (e.g., abc123xy)')
    parser.add_argument('--wandb_project', type=str, help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity name')
    parser.add_argument('--log_file', type=str, help='Path to CSV log file')
    parser.add_argument('--output_dir', type=str, default='loss_plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    if args.wandb_run:
        data = load_from_wandb(args.wandb_run, args.wandb_project, args.wandb_entity)
    elif args.log_file:
        data = load_from_csv(args.log_file)
    else:
        print("Error: Must provide either --wandb_run or --log_file")
        return
    
    if data is None or len(data) == 0:
        print("Error: No data loaded!")
        return
    
    print(f"Loaded {len(data)} steps of training data")
    print(f"Columns: {list(data.columns)}")
    
    # Plot
    plot_action_losses(data, output_dir=args.output_dir)
    
    print(f"\nDone! Check {args.output_dir}/ for plots.")


if __name__ == "__main__":
    main()
