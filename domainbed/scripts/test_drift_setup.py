#!/usr/bin/env python
"""
Quick test to verify drift tracking works before running full experiment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import json
from collections import deque
import numpy as np

# Test geomloss
try:
    import geomloss
    print("✓ geomloss imported successfully")
except ImportError:
    print("✗ geomloss not installed. Run: pip install geomloss")
    sys.exit(1)

# Test domainbed imports
try:
    from domainbed import algorithms, datasets, hparams_registry
    print("✓ domainbed imported successfully")
except Exception as e:
    print(f"✗ domainbed import failed: {e}")
    sys.exit(1)

# Test SODA algorithm exists
try:
    soda_class = algorithms.get_algorithm_class('SODA')
    print(f"✓ SODA algorithm found: {soda_class}")
except Exception as e:
    print(f"✗ SODA algorithm not found: {e}")
    sys.exit(1)

# Test ERM algorithm
try:
    erm_class = algorithms.get_algorithm_class('ERM')
    print(f"✓ ERM algorithm found: {erm_class}")
except Exception as e:
    print(f"✗ ERM algorithm not found: {e}")
    sys.exit(1)

# Test drift tracker
try:
    from domainbed.scripts.compute_target_drift import TargetDriftTracker
    
    tracker = TargetDriftTracker(method='soda', reservoir_size=64, feature_dim=128)
    
    # Simulate some batches
    for i in range(10):
        fake_features = torch.randn(16, 128)
        drift = tracker.compute_drift(fake_features, i * 10)
    
    print(f"✓ TargetDriftTracker works. Sample drifts: {tracker.drift_history[:3]}")
except Exception as e:
    print(f"✗ TargetDriftTracker failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test W2 computation with geomloss
try:
    w2_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.1)
    x = torch.randn(32, 128)
    y = torch.randn(64, 128)
    loss = w2_loss(x, y)
    print(f"✓ Sinkhorn loss works. Sample loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Sinkhorn loss failed: {e}")
    sys.exit(1)

# Check GPUs
print(f"\n✓ All tests passed!")
print(f"  Available GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

print("\nYou can now run the full experiment:")
print("  cd /home/mosama97/UBCO/research/SODA/domainbed/scripts")
print("  bash run_drift_experiment.sh")
