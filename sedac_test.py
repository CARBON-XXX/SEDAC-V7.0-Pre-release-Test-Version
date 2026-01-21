"""
SEDAC Unified Test Framework
=============================

Complete test suite for SEDAC acceleration validation.

Test Methodology:
================

1. DATA COLLECTION
   - Collect hidden states from ALL layers during real inference
   - Record token-level output entropy as ground truth
   - No simulation - uses actual model outputs

2. EXIT SIMULATION
   - Replay collected hidden states through SEDAC monitors
   - Record which layer each token would exit at
   - Compare exit decision to ground truth entropy

3. QUALITY METRICS
   - High Risk Rate: % of early-exit tokens with high final entropy
   - PPL Loss: Entropy increase due to early exit
   - Exit Distribution: Which layers are used for exit

4. SPEEDUP CALCULATION
   - Speedup = total_layers / avg_exit_layer
   - NOT simulated timing - based on layer computation ratio

Usage:
    # Full test with auto-calibration
    python sedac_test.py --data-dir sedac_data --auto-calibrate
    
    # Test specific V7 config
    python sedac_test.py --data-dir sedac_data --v7 --tau 0.97 --k 2
    
    # Test specific V6 config
    python sedac_test.py --data-dir sedac_data --v6 --threshold-scale 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from sedac.core.universal_monitor import UniversalStabilityMonitor, UniversalConfig
from sedac.core.probe_inference import LREProbe


@dataclass
class TestResult:
    """Complete test result with all metrics."""
    method: str
    config: Dict
    
    # Performance
    speedup: float
    exit_rate: float
    avg_exit_layer: float
    
    # Quality
    high_risk_rate: float
    ppl_loss_pct: float
    
    # Distribution
    exit_distribution: Dict[int, int]
    
    def summary(self) -> str:
        return (
            f"Method: {self.method}\n"
            f"Speedup: {self.speedup:.2f}x\n"
            f"Exit Rate: {self.exit_rate*100:.1f}%\n"
            f"High Risk: {self.high_risk_rate*100:.1f}%\n"
            f"PPL Loss: {self.ppl_loss_pct:+.2f}%"
        )


class SEDACTester:
    """
    Unified SEDAC Test Framework.
    
    Test Flow:
    1. Load pre-collected hidden states and entropies
    2. Simulate SEDAC exit decisions
    3. Compare to ground truth (final layer entropy)
    4. Compute speedup and quality metrics
    """
    
    def __init__(self, data_dir: str, device: str = "cuda"):
        self.data_dir = Path(data_dir)
        self.device = device
        
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.entropies: Dict[int, torch.Tensor] = {}
        self.probes: Dict[int, LREProbe] = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load all layer data."""
        print("=" * 60)
        print("SEDAC Test Framework")
        print("=" * 60)
        print(f"\nLoading data from: {self.data_dir}")
        
        for layer_idx in range(50):
            h_path = self.data_dir / f"hidden_states_layer{layer_idx}.pt"
            e_path = self.data_dir / f"entropies_layer{layer_idx}.pt"
            
            if h_path.exists() and e_path.exists():
                self.hidden_states[layer_idx] = torch.load(h_path, map_location="cpu").float()
                self.entropies[layer_idx] = torch.load(e_path, map_location="cpu").float()
        
        self.available_layers = sorted(self.hidden_states.keys())
        self.total_layers = max(self.available_layers) + 1
        self.num_samples = self.hidden_states[self.available_layers[0]].shape[0]
        self.final_layer = self.available_layers[-1]
        
        print(f"  Layers: {len(self.available_layers)} ({self.available_layers[0]}-{self.final_layer})")
        print(f"  Samples: {self.num_samples}")
        
        # Load probes
        for layer_idx in self.available_layers:
            probe_path = self.data_dir / f"sedac_probe_layer{layer_idx}.pth"
            if probe_path.exists():
                try:
                    hidden_dim = self.hidden_states[layer_idx].shape[1]
                    probe = LREProbe(hidden_dim, rank=64)
                    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
                    probe.to(self.device).eval()
                    self.probes[layer_idx] = probe
                except Exception:
                    pass
        
        if self.probes:
            print(f"  V6 Probes: {len(self.probes)} layers")
        
        # Compute baseline metrics
        self.baseline_entropy = self.entropies[self.final_layer].mean().item()
        self.entropy_75th = self.entropies[self.final_layer].quantile(0.75).item()
        
        print(f"\n  Baseline entropy: {self.baseline_entropy:.4f}")
        print(f"  High-risk threshold (75th): {self.entropy_75th:.4f}")
    
    def test_v7(
        self,
        tau: float,
        consecutive_k: int,
        min_layer_ratio: float = 0.15,
    ) -> TestResult:
        """Test V7 universal stability monitor."""
        config = UniversalConfig(
            tau=tau,
            consecutive_k=consecutive_k,
            min_layer_ratio=min_layer_ratio,
        )
        
        exit_layers = []
        exit_entropies = []
        
        for sample_idx in range(self.num_samples):
            monitor = UniversalStabilityMonitor(config)
            exit_layer = self.final_layer
            
            for layer_idx in self.available_layers:
                h = self.hidden_states[layer_idx][sample_idx:sample_idx+1]
                if monitor.step(h, layer_idx, self.total_layers):
                    exit_layer = layer_idx
                    break
            
            exit_layers.append(exit_layer)
            exit_entropies.append(self.entropies[exit_layer][sample_idx].item())
        
        return self._compute_metrics(
            "v7",
            {"tau": tau, "K": consecutive_k},
            exit_layers,
            exit_entropies,
        )
    
    def test_v6(
        self,
        threshold_scale: float = 1.0,
        thresholds: Optional[Dict[int, float]] = None,
    ) -> TestResult:
        """Test V6 probe-based exit."""
        if not self.probes:
            raise ValueError("No V6 probes loaded")
        
        probe_layers = sorted(self.probes.keys())
        
        # Compute thresholds if not provided
        if thresholds is None:
            thresholds = {}
            for layer_idx in probe_layers:
                h = self.hidden_states[layer_idx].to(self.device)
                with torch.inference_mode():
                    risks = self.probes[layer_idx](h).squeeze(-1)
                thresholds[layer_idx] = risks.quantile(0.5).item() * threshold_scale
        
        exit_layers = []
        exit_entropies = []
        
        for sample_idx in range(self.num_samples):
            exit_layer = self.final_layer
            
            for layer_idx in probe_layers:
                h = self.hidden_states[layer_idx][sample_idx:sample_idx+1].to(self.device)
                with torch.inference_mode():
                    risk = self.probes[layer_idx](h).item()
                
                if risk < thresholds[layer_idx]:
                    exit_layer = layer_idx
                    break
            
            exit_layers.append(exit_layer)
            exit_entropies.append(self.entropies[exit_layer][sample_idx].item())
        
        return self._compute_metrics(
            "v6",
            {"threshold_scale": threshold_scale},
            exit_layers,
            exit_entropies,
        )
    
    def _compute_metrics(
        self,
        method: str,
        config: Dict,
        exit_layers: List[int],
        exit_entropies: List[float],
    ) -> TestResult:
        """Compute all test metrics."""
        # Basic metrics
        avg_exit_layer = sum(exit_layers) / len(exit_layers)
        speedup = self.total_layers / avg_exit_layer if avg_exit_layer > 0 else 1.0
        exit_rate = sum(1 for e in exit_layers if e < self.final_layer) / len(exit_layers)
        
        # High risk rate
        high_risk = 0
        final_ents = self.entropies[self.final_layer]
        for i, el in enumerate(exit_layers):
            if el < self.final_layer and final_ents[i].item() > self.entropy_75th:
                high_risk += 1
        high_risk_rate = high_risk / len(exit_layers)
        
        # PPL loss (entropy-based proxy)
        avg_exit_entropy = sum(exit_entropies) / len(exit_entropies)
        ppl_loss_pct = (avg_exit_entropy - self.baseline_entropy) / self.baseline_entropy * 100
        
        # Exit distribution
        exit_dist = {}
        for el in exit_layers:
            exit_dist[el] = exit_dist.get(el, 0) + 1
        
        return TestResult(
            method=method,
            config=config,
            speedup=speedup,
            exit_rate=exit_rate,
            avg_exit_layer=avg_exit_layer,
            high_risk_rate=high_risk_rate,
            ppl_loss_pct=ppl_loss_pct,
            exit_distribution=exit_dist,
        )
    
    def run_full_test(self) -> Dict[str, TestResult]:
        """Run comprehensive test suite."""
        results = {}
        
        print("\n" + "=" * 60)
        print("RUNNING FULL TEST SUITE")
        print("=" * 60)
        
        # V7 tests
        print("\n--- V7 Universal Monitor ---")
        v7_configs = [
            (0.99, 3, "conservative"),
            (0.98, 2, "balanced"),
            (0.97, 2, "moderate"),
            (0.95, 2, "aggressive"),
        ]
        
        for tau, k, name in v7_configs:
            result = self.test_v7(tau, k)
            results[f"v7_{name}"] = result
            print(f"  [{name}] tau={tau}, K={k}: "
                  f"{result.speedup:.2f}x speedup, "
                  f"{result.high_risk_rate*100:.1f}% risk, "
                  f"{result.ppl_loss_pct:+.2f}% PPL")
        
        # V6 tests (if available)
        if self.probes:
            print("\n--- V6 Probe-Based ---")
            v6_scales = [0.8, 1.0, 1.2]
            
            for scale in v6_scales:
                result = self.test_v6(scale)
                results[f"v6_scale{scale}"] = result
                print(f"  [scale={scale}]: "
                      f"{result.speedup:.2f}x speedup, "
                      f"{result.high_risk_rate*100:.1f}% risk, "
                      f"{result.ppl_loss_pct:+.2f}% PPL")
        
        return results
    
    def print_methodology(self):
        """Print test methodology explanation."""
        print("""
================================================================================
SEDAC TEST METHODOLOGY
================================================================================

1. DATA COLLECTION PHASE
   - Run real inference on calibration dataset
   - Hook all transformer layers to capture hidden states
   - Record output entropy for each token (ground truth)
   
2. EXIT SIMULATION PHASE
   - Replay saved hidden states through SEDAC monitors
   - V6: Check if probe(hidden) < threshold at each checkpoint
   - V7: Check if stability(h_l, h_{l-1}) >= tau for K consecutive layers
   - Record exit layer for each token
   
3. QUALITY EVALUATION
   - Compare exit entropy to final-layer entropy
   - High Risk = tokens that exit early but have high final entropy
   - PPL Loss = (avg_exit_entropy - baseline_entropy) / baseline_entropy
   
4. SPEEDUP CALCULATION
   - Speedup = total_layers / avg_exit_layer
   - This represents compute savings, NOT wall-clock time
   - Actual inference speedup depends on implementation efficiency

KEY INSIGHT:
- Negative PPL Loss means early-exit tokens have LOWER entropy
- This happens because SEDAC selects "easy" tokens for early exit
- Easy tokens naturally have lower uncertainty

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="SEDAC Unified Test Framework")
    parser.add_argument("--data-dir", type=str, default="sedac_data")
    parser.add_argument("--methodology", action="store_true",
                        help="Print test methodology and exit")
    parser.add_argument("--full", action="store_true",
                        help="Run full test suite")
    parser.add_argument("--v7", action="store_true", help="Test V7")
    parser.add_argument("--v6", action="store_true", help="Test V6")
    parser.add_argument("--tau", type=float, default=0.97)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    args = parser.parse_args()
    
    tester = SEDACTester(args.data_dir)
    
    if args.methodology:
        tester.print_methodology()
        return 0
    
    if args.full:
        results = tester.run_full_test()
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for name, result in results.items():
            print(f"\n[{name}]")
            print(result.summary())
        return 0
    
    if args.v7:
        result = tester.test_v7(args.tau, args.k)
        print("\n" + result.summary())
    
    if args.v6:
        result = tester.test_v6(args.threshold_scale)
        print("\n" + result.summary())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
