"""
Training script that logs target drift during training.
Run for both ERM and SODA to generate comparison data for Figure 5.

Usage:
    # Run both algorithms (uses 2 GPUs in parallel)
    python train_drift_comparison.py --run_both --dataset OfficeHome
    
    # Run single algorithm
    python train_drift_comparison.py --algorithm ERM --gpu 0
    python train_drift_comparison.py --algorithm SODA --gpu 1
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm

from domainbed import datasets, hparams_registry, algorithms
from domainbed.lib.fast_data_loader import InfiniteDataLoader
from domainbed.scripts.compute_target_drift import TargetDriftTracker


def train_with_drift_logging(
    algorithm_name: str,
    dataset_name: str = "OfficeHome",
    test_env: int = 0,
    num_iterations: int = 5000,
    log_every: int = 10,
    output_dir: str = "./drift_results",
    gpu_id: int = 0,
    data_dir: str = None
):
    """
    Train model and log target drift.
    
    Args:
        algorithm_name: 'ERM' or 'SODA'
        dataset_name: Dataset to use
        test_env: Index of test environment
        num_iterations: Total training iterations
        log_every: Log drift every N iterations
        output_dir: Where to save results
        gpu_id: Which GPU to use
        data_dir: Path to data directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    
    print(f"[{algorithm_name}] Using device: {device}")
    
    # Setup dataset
    hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)
    
    # Override some hparams for faster training if needed
    hparams['batch_size'] = 32
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print(f"[{algorithm_name}] Loading dataset from: {data_dir}")
    
    dataset = vars(datasets)[dataset_name](data_dir, [test_env], hparams)
    
    # Get training environments (exclude test)
    train_envs = [i for i in range(len(dataset)) if i != test_env]
    print(f"[{algorithm_name}] Training on environments: {train_envs}, testing on: {test_env}")
    
    train_loaders = [
        InfiniteDataLoader(
            dataset=dataset[i],
            weights=None,
            batch_size=hparams['batch_size'],
            num_workers=2
        )
        for i in train_envs
    ]
    train_iters = [iter(loader) for loader in train_loaders]
    
    # Initialize algorithm
    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(train_envs),
        hparams
    ).to(device)
    
    # Initialize drift tracker
    # For ERM: track batch-derived pivot (small, unstable)
    # For SODA: track reservoir pivot (large, stable)
    method = 'erm' if algorithm_name == 'ERM' else 'soda'
    reservoir_size = hparams.get('ema_pivot_size', 256) if method == 'soda' else hparams['batch_size']
    
    # Get feature dimension
    if hasattr(algorithm, 'featurizer'):
        feature_dim = algorithm.featurizer.n_outputs
    else:
        feature_dim = 2048  # Default for ResNet-50
    
    drift_tracker = TargetDriftTracker(
        method=method,
        reservoir_size=reservoir_size,
        feature_dim=feature_dim,
        device=str(device)
    )
    
    print(f"[{algorithm_name}] Training for {num_iterations} iterations...")
    print(f"[{algorithm_name}] Tracking {method} pivot drift (reservoir_size={reservoir_size}, feature_dim={feature_dim})")
    
    # Training loop
    pbar = tqdm(range(num_iterations), desc=f"{algorithm_name}", position=gpu_id)
    for iteration in pbar:
        # Get minibatches from all training domains
        minibatches = [next(train_iter) for train_iter in train_iters]
        minibatches = [(x.to(device), y.to(device)) for x, y in minibatches]
        
        # Compute drift (every log_every iterations)
        if iteration % log_every == 0:
            with torch.no_grad():
                all_x = torch.cat([x for x, y in minibatches])
                if hasattr(algorithm, 'featurizer'):
                    features = algorithm.featurizer(all_x)
                elif hasattr(algorithm, 'network') and hasattr(algorithm.network, '__getitem__'):
                    features = algorithm.network[0](all_x)
                else:
                    # Fallback: just use first part of network
                    features = algorithm.network(all_x)
                
                drift = drift_tracker.compute_drift(features, iteration)
        
        # Actual training step
        step_results = algorithm.update(minibatches)
        
        # Update progress bar
        if iteration % 50 == 0:
            loss_val = step_results.get('loss', step_results.get('loss_cls', 0))
            drift_val = drift_tracker.drift_history[-1] if drift_tracker.drift_history else 0
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'drift': f'{drift_val:.4f}'
            })
    
    # Save results
    output_file = os.path.join(output_dir, f"drift_{algorithm_name.lower()}_{dataset_name.lower()}.json")
    drift_tracker.save_results(output_file)
    
    print(f"[{algorithm_name}] Completed! Results saved to {output_file}")
    return output_file


def run_algorithm(args_tuple):
    """Wrapper for multiprocessing."""
    algorithm, dataset, test_env, num_iters, log_every, output_dir, gpu_id, data_dir = args_tuple
    try:
        return train_with_drift_logging(
            algorithm_name=algorithm,
            dataset_name=dataset,
            test_env=test_env,
            num_iterations=num_iters,
            log_every=log_every,
            output_dir=output_dir,
            gpu_id=gpu_id,
            data_dir=data_dir
        )
    except Exception as e:
        print(f"Error training {algorithm}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Train with target drift logging")
    parser.add_argument('--algorithm', type=str, default='SODA', 
                        choices=['ERM', 'SODA'], help='Algorithm to train')
    parser.add_argument('--dataset', type=str, default='OfficeHome',
                        help='Dataset name')
    parser.add_argument('--test_env', type=int, default=0,
                        help='Test environment index')
    parser.add_argument('--iterations', type=int, default=5000,
                        help='Number of training iterations')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log drift every N iterations')
    parser.add_argument('--output_dir', type=str, default='./drift_results',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--run_both', action='store_true',
                        help='Run both ERM and SODA in parallel on 2 GPUs')
    
    args = parser.parse_args()
    
    # Resolve output directory
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(__file__), '..', '..', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve data directory
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    
    if args.run_both:
        # Run both algorithms in parallel using 2 GPUs
        print("Running ERM on GPU 0 and SODA on GPU 1 in parallel...")
        
        # Prepare arguments for each process
        erm_args = ('ERM', args.dataset, args.test_env, args.iterations, 
                    args.log_every, args.output_dir, 0, args.data_dir)
        soda_args = ('SODA', args.dataset, args.test_env, args.iterations,
                     args.log_every, args.output_dir, 1, args.data_dir)
        
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context('spawn')
        
        with ctx.Pool(2) as pool:
            results = pool.map(run_algorithm, [erm_args, soda_args])
        
        print(f"\nResults saved to:")
        for r in results:
            if r:
                print(f"  - {r}")
    else:
        # Run single algorithm
        train_with_drift_logging(
            algorithm_name=args.algorithm,
            dataset_name=args.dataset,
            test_env=args.test_env,
            num_iterations=args.iterations,
            log_every=args.log_every,
            output_dir=args.output_dir,
            gpu_id=args.gpu,
            data_dir=args.data_dir
        )


if __name__ == "__main__":
    main()
