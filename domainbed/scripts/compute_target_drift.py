"""
Compute Target Drift for SODA vs ERM baseline.
Measures W2 distance between consecutive alignment targets.
"""

import torch
import torch.nn.functional as F
import numpy as np
import geomloss
from collections import deque
import json


class TargetDriftTracker:
    """Tracks the Wasserstein-2 distance between consecutive pivot distributions."""
    
    def __init__(self, method: str, reservoir_size: int = 256, feature_dim: int = 2048, device: str = "cuda"):
        """
        Args:
            method: 'erm' (batch-derived pivot) or 'soda' (reservoir pivot)
            reservoir_size: Size K of the reservoir (for SODA)
            feature_dim: Dimension d' of features
            device: Device to use for computation
        """
        self.method = method
        self.reservoir_size = reservoir_size
        self.feature_dim = feature_dim
        self.device = device
        
        # For computing W2 distance using Sinkhorn
        self.w2_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1, scaling=0.9)
        
        # Store previous pivot for comparison
        self.prev_pivot = None
        
        # For SODA: maintain a reservoir queue (store on CPU to save GPU memory)
        if method == 'soda':
            self.reservoir = deque(maxlen=reservoir_size)
        
        # Store drift values over training
        self.drift_history = []
        self.iteration_history = []
    
    def compute_drift(self, batch_features: torch.Tensor, iteration: int) -> float:
        """
        Compute target drift for current iteration.
        
        Args:
            batch_features: Current batch features [B, d']
            iteration: Current training iteration
            
        Returns:
            W2 distance between current and previous pivot (0.0 if first iteration)
        """
        batch_features = batch_features.detach()
        batch_features_normalized = F.normalize(batch_features, dim=1)
        
        if self.method == 'erm':
            # ERM: pivot is just the current batch
            current_pivot = batch_features_normalized
        else:
            # SODA: pivot is the reservoir
            # First, update reservoir with current batch
            for feat in batch_features_normalized:
                self.reservoir.append(feat.cpu())
            
            # Current pivot is the entire reservoir
            if len(self.reservoir) < self.reservoir_size // 4:
                # Not enough samples yet, skip
                return 0.0
            
            current_pivot = torch.stack(list(self.reservoir)).to(self.device)
        
        # Compute W2 distance from previous pivot
        drift = 0.0
        if self.prev_pivot is not None and current_pivot.shape[0] > 1:
            try:
                # Ensure both are on same device
                prev_on_device = self.prev_pivot.to(self.device)
                current_on_device = current_pivot.to(self.device)
                
                # For fair comparison, subsample if sizes differ significantly
                min_size = min(prev_on_device.shape[0], current_on_device.shape[0])
                if prev_on_device.shape[0] > min_size:
                    indices = torch.randperm(prev_on_device.shape[0])[:min_size]
                    prev_on_device = prev_on_device[indices]
                if current_on_device.shape[0] > min_size:
                    indices = torch.randperm(current_on_device.shape[0])[:min_size]
                    current_on_device = current_on_device[indices]
                
                drift = self.w2_loss(current_on_device, prev_on_device).item()
            except Exception as e:
                print(f"Warning: W2 computation failed at iter {iteration}: {e}")
                drift = 0.0
        
        # Update previous pivot (store on CPU)
        self.prev_pivot = current_pivot.cpu().clone()
        
        # Record history
        self.drift_history.append(drift)
        self.iteration_history.append(iteration)
        
        return drift
    
    def save_results(self, filepath: str):
        """Save drift history to JSON file."""
        results = {
            'method': self.method,
            'reservoir_size': self.reservoir_size,
            'iterations': self.iteration_history,
            'drift_values': self.drift_history
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved drift results to {filepath}")
