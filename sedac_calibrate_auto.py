"""
SEDAC Adaptive Auto-Calibration System
=======================================

Automatically finds optimal SEDAC configuration based on user constraints.

Key Features:
1. Pareto-optimal search: Finds best speedup given quality constraints
2. Zero manual tuning: User only specifies acceptable risk level
3. Unified V6/V7 calibration: Works with both probe-based and universal methods

Calibration Modes:
- Quality-first: Maximize speedup while keeping high_risk_rate < threshold
- Speed-first: Maximize speedup while keeping min_layer > safety_floor
- Balanced: Pareto-optimal trade-off between speedup and quality

Usage:
    python sedac_calibrate_auto.py --mode balanced --max-risk 0.15
    python sedac_calibrate_auto.py --mode quality --max-risk 0.10
    python sedac_calibrate_auto.py --mode speed --min-speedup 3.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from sedac.core.universal_monitor import UniversalStabilityMonitor, UniversalConfig
from sedac.core.probe_inference import LREProbe


@dataclass
class CalibrationResult:
    """Result of a single calibration run."""
    method: str  # "v6" or "v7"
    config: Dict
    speedup: float
    exit_rate: float
    avg_exit_layer: float
    high_risk_rate: float
    quality_score: float  # 1.0 - high_risk_rate
    
    def __repr__(self):
        return (f"CalibrationResult(method={self.method}, speedup={self.speedup:.2f}x, "
                f"exit_rate={self.exit_rate*100:.1f}%, risk={self.high_risk_rate*100:.1f}%)")


@dataclass
class OptimalConfig:
    """Optimal configuration found by calibration."""
    method: str
    speedup: float
    exit_rate: float
    high_risk_rate: float
    
    # V6 config
    v6_thresholds: Optional[Dict[int, float]] = None
    
    # V7 config
    v7_tau: Optional[float] = None
    v7_consecutive_k: Optional[int] = None
    v7_min_layer_ratio: Optional[float] = None
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert to environment variables for SEDAC runtime."""
        env = {"SEDAC_METHOD": self.method}
        
        if self.method == "v6" and self.v6_thresholds:
            env["SEDAC_V6_LAYERS"] = ",".join(str(k) for k in sorted(self.v6_thresholds.keys()))
            env["SEDAC_V6_THRESHOLDS"] = ",".join(f"{v:.4f}" for k, v in sorted(self.v6_thresholds.items()))
        
        if self.method == "v7":
            env["SEDAC_V7_TAU"] = str(self.v7_tau)
            env["SEDAC_V7_K"] = str(self.v7_consecutive_k)
            env["SEDAC_V7_MIN_LAYER"] = str(self.v7_min_layer_ratio)
        
        return env
    
    def to_json(self) -> str:
        """Serialize to JSON for persistence."""
        return json.dumps(asdict(self), indent=2)


class AdaptiveCalibrator:
    """
    SEDAC Adaptive Calibration System.
    
    Automatically finds optimal configuration through grid search
    with Pareto-optimal selection.
    """
    
    def __init__(
        self,
        data_dir: str = "sedac_data",
        device: str = "cuda",
        num_samples: int = 1000,
    ):
        self.data_dir = Path(data_dir)
        self.device = device
        self.num_samples = num_samples
        
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.entropies: Dict[int, torch.Tensor] = {}
        self.probes: Dict[int, LREProbe] = {}
        
        self._loaded = False
    
    def load_data(self) -> bool:
        """Load calibration data."""
        print("Loading calibration data...")
        
        # Load hidden states and entropies
        for layer_idx in range(50):  # Support up to 50 layers
            h_path = self.data_dir / f"hidden_states_layer{layer_idx}.pt"
            e_path = self.data_dir / f"entropies_layer{layer_idx}.pt"
            
            if h_path.exists() and e_path.exists():
                h = torch.load(h_path, map_location="cpu")[:self.num_samples].float()
                e = torch.load(e_path, map_location="cpu")[:self.num_samples].float()
                self.hidden_states[layer_idx] = h
                self.entropies[layer_idx] = e
        
        if not self.hidden_states:
            print("  [ERROR] No data found in", self.data_dir)
            return False
        
        available_layers = sorted(self.hidden_states.keys())
        print(f"  Loaded {len(available_layers)} layers: {available_layers[:5]}...{available_layers[-3:]}")
        print(f"  Samples: {self.hidden_states[available_layers[0]].shape[0]}")
        
        # Load V6 probes if available
        for layer_idx in available_layers:
            probe_path = self.data_dir / f"sedac_probe_layer{layer_idx}.pth"
            if probe_path.exists():
                try:
                    probe = LREProbe(self.hidden_states[layer_idx].shape[1], rank=64)
                    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
                    probe.to(self.device).eval()
                    self.probes[layer_idx] = probe
                except Exception as e:
                    pass
        
        if self.probes:
            print(f"  Loaded {len(self.probes)} V6 probes: {sorted(self.probes.keys())}")
        
        self._loaded = True
        return True
    
    def _compute_high_risk_rate(
        self,
        exit_layers: List[int],
        exit_entropies: List[float],
    ) -> float:
        """Compute high-risk rate (exits with high final entropy)."""
        sorted_layers = sorted(self.entropies.keys())
        final_layer = sorted_layers[-1]
        final_entropies = self.entropies[final_layer]
        
        # 75th percentile as high-risk threshold
        entropy_threshold = final_entropies.quantile(0.75).item()
        
        high_risk = 0
        for i, (el, _) in enumerate(zip(exit_layers, exit_entropies)):
            if el < final_layer:
                if i < len(final_entropies) and final_entropies[i].item() > entropy_threshold:
                    high_risk += 1
        
        return high_risk / len(exit_layers) if exit_layers else 0
    
    def calibrate_v7(
        self,
        tau: float,
        consecutive_k: int,
        min_layer_ratio: float = 0.15,
    ) -> CalibrationResult:
        """Run calibration for a V7 configuration."""
        config = UniversalConfig(
            tau=tau,
            consecutive_k=consecutive_k,
            min_layer_ratio=min_layer_ratio,
        )
        
        sorted_layers = sorted(self.hidden_states.keys())
        total_layers = max(sorted_layers) + 1
        num_samples = min(self.num_samples, self.hidden_states[sorted_layers[0]].shape[0])
        
        exit_layers = []
        exit_entropies = []
        
        for sample_idx in range(num_samples):
            monitor = UniversalStabilityMonitor(config)
            exit_layer = sorted_layers[-1]
            
            for layer_idx in sorted_layers:
                h = self.hidden_states[layer_idx][sample_idx:sample_idx+1]
                should_exit = monitor.step(h, layer_idx, total_layers)
                
                if should_exit:
                    exit_layer = layer_idx
                    break
            
            exit_layers.append(exit_layer)
            if exit_layer in self.entropies:
                exit_entropies.append(self.entropies[exit_layer][sample_idx].item())
            else:
                exit_entropies.append(0)
        
        avg_exit_layer = sum(exit_layers) / len(exit_layers)
        speedup = total_layers / avg_exit_layer if avg_exit_layer > 0 else 1.0
        exit_rate = sum(1 for e in exit_layers if e < sorted_layers[-1]) / len(exit_layers)
        high_risk_rate = self._compute_high_risk_rate(exit_layers, exit_entropies)
        
        return CalibrationResult(
            method="v7",
            config={"tau": tau, "K": consecutive_k, "min_layer_ratio": min_layer_ratio},
            speedup=speedup,
            exit_rate=exit_rate,
            avg_exit_layer=avg_exit_layer,
            high_risk_rate=high_risk_rate,
            quality_score=1.0 - high_risk_rate,
        )
    
    def calibrate_v6(
        self,
        threshold_scale: float,
    ) -> CalibrationResult:
        """Run calibration for a V6 configuration with scaled thresholds."""
        if not self.probes:
            return CalibrationResult(
                method="v6", config={}, speedup=1.0, exit_rate=0,
                avg_exit_layer=36, high_risk_rate=0, quality_score=1.0
            )
        
        sorted_layers = sorted(self.hidden_states.keys())
        probe_layers = sorted(self.probes.keys())
        total_layers = max(sorted_layers) + 1
        num_samples = min(self.num_samples, self.hidden_states[sorted_layers[0]].shape[0])
        
        # Compute thresholds based on scale
        thresholds = {}
        for layer_idx in probe_layers:
            h = self.hidden_states[layer_idx].to(self.device)
            with torch.inference_mode():
                risks = self.probes[layer_idx](h).squeeze(-1)
            base_thr = risks.quantile(0.5).item()
            thresholds[layer_idx] = base_thr * threshold_scale
        
        # Simulate exits
        exit_layers = []
        exit_entropies = []
        
        for sample_idx in range(num_samples):
            exit_layer = sorted_layers[-1]
            
            for layer_idx in probe_layers:
                h = self.hidden_states[layer_idx][sample_idx:sample_idx+1].to(self.device)
                with torch.inference_mode():
                    risk = self.probes[layer_idx](h).item()
                
                if risk < thresholds[layer_idx]:
                    exit_layer = layer_idx
                    break
            
            exit_layers.append(exit_layer)
            if exit_layer in self.entropies:
                exit_entropies.append(self.entropies[exit_layer][sample_idx].item())
            else:
                exit_entropies.append(0)
        
        avg_exit_layer = sum(exit_layers) / len(exit_layers)
        speedup = total_layers / avg_exit_layer if avg_exit_layer > 0 else 1.0
        exit_rate = sum(1 for e in exit_layers if e < sorted_layers[-1]) / len(exit_layers)
        high_risk_rate = self._compute_high_risk_rate(exit_layers, exit_entropies)
        
        return CalibrationResult(
            method="v6",
            config={"threshold_scale": threshold_scale, "thresholds": thresholds},
            speedup=speedup,
            exit_rate=exit_rate,
            avg_exit_layer=avg_exit_layer,
            high_risk_rate=high_risk_rate,
            quality_score=1.0 - high_risk_rate,
        )
    
    def find_optimal(
        self,
        mode: str = "balanced",
        max_risk: float = 0.15,
        min_speedup: float = 1.5,
        target_speedup: Optional[float] = None,
    ) -> OptimalConfig:
        """
        Find optimal SEDAC configuration.
        
        Args:
            mode: "quality" | "balanced" | "speed"
            max_risk: Maximum acceptable high-risk rate
            min_speedup: Minimum acceptable speedup
            target_speedup: If set, find config closest to this speedup
        
        Returns:
            OptimalConfig with best settings
        """
        if not self._loaded:
            self.load_data()
        
        print(f"\nCalibrating SEDAC (mode={mode}, max_risk={max_risk}, min_speedup={min_speedup})")
        print("=" * 60)
        
        all_results: List[CalibrationResult] = []
        
        # V7 grid search
        print("\n--- V7 Calibration ---")
        tau_range = [0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.95, 0.94, 0.92, 0.90]
        k_range = [2, 3]
        
        for tau in tau_range:
            for k in k_range:
                result = self.calibrate_v7(tau, k)
                all_results.append(result)
                
                if result.speedup >= min_speedup:
                    mark = "[*]" if result.high_risk_rate <= max_risk else "[ ]"
                    print(f"  {mark} tau={tau}, K={k}: {result.speedup:.2f}x, risk={result.high_risk_rate*100:.1f}%")
        
        # V6 grid search (if probes available)
        if self.probes:
            print("\n--- V6 Calibration ---")
            scale_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
            
            for scale in scale_range:
                result = self.calibrate_v6(scale)
                all_results.append(result)
                
                if result.speedup >= min_speedup:
                    mark = "[*]" if result.high_risk_rate <= max_risk else "[ ]"
                    print(f"  {mark} scale={scale}: {result.speedup:.2f}x, risk={result.high_risk_rate*100:.1f}%")
        
        # Filter by constraints
        valid_results = [
            r for r in all_results
            if r.high_risk_rate <= max_risk and r.speedup >= min_speedup
        ]
        
        if not valid_results:
            print("\n[WARN] No config satisfies constraints. Relaxing...")
            valid_results = [r for r in all_results if r.speedup >= 1.0]
            valid_results.sort(key=lambda r: r.high_risk_rate)
            valid_results = valid_results[:5]
        
        # Select best based on mode
        if mode == "quality":
            # Minimize risk, then maximize speedup
            valid_results.sort(key=lambda r: (r.high_risk_rate, -r.speedup))
        elif mode == "speed":
            # Maximize speedup, then minimize risk
            valid_results.sort(key=lambda r: (-r.speedup, r.high_risk_rate))
        elif mode == "balanced":
            # Pareto-optimal: maximize (speedup * quality_score)
            valid_results.sort(key=lambda r: -(r.speedup * r.quality_score))
        elif target_speedup is not None:
            # Find closest to target speedup
            valid_results.sort(key=lambda r: abs(r.speedup - target_speedup))
        
        best = valid_results[0]
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL CONFIG FOUND")
        print(f"{'='*60}")
        print(f"  Method: {best.method.upper()}")
        print(f"  Speedup: {best.speedup:.2f}x")
        print(f"  Exit Rate: {best.exit_rate*100:.1f}%")
        print(f"  High Risk: {best.high_risk_rate*100:.1f}%")
        print(f"  Config: {best.config}")
        
        # Build OptimalConfig
        if best.method == "v7":
            optimal = OptimalConfig(
                method="v7",
                speedup=best.speedup,
                exit_rate=best.exit_rate,
                high_risk_rate=best.high_risk_rate,
                v7_tau=best.config["tau"],
                v7_consecutive_k=best.config["K"],
                v7_min_layer_ratio=best.config.get("min_layer_ratio", 0.15),
            )
        else:
            optimal = OptimalConfig(
                method="v6",
                speedup=best.speedup,
                exit_rate=best.exit_rate,
                high_risk_rate=best.high_risk_rate,
                v6_thresholds=best.config.get("thresholds"),
            )
        
        return optimal


def main():
    parser = argparse.ArgumentParser(description="SEDAC Adaptive Auto-Calibration")
    parser.add_argument("--data-dir", type=str, default="sedac_data")
    parser.add_argument("--mode", type=str, default="balanced", 
                        choices=["quality", "balanced", "speed"])
    parser.add_argument("--max-risk", type=float, default=0.15,
                        help="Maximum acceptable high-risk rate (default: 0.15)")
    parser.add_argument("--min-speedup", type=float, default=1.5,
                        help="Minimum acceptable speedup (default: 1.5)")
    parser.add_argument("--target-speedup", type=float, default=None,
                        help="Target speedup to achieve")
    parser.add_argument("--output", type=str, default="sedac_config.json",
                        help="Output config file")
    args = parser.parse_args()
    
    calibrator = AdaptiveCalibrator(data_dir=args.data_dir)
    
    if not calibrator.load_data():
        print("Failed to load data. Run sedac_collect_data.py first.")
        return 1
    
    optimal = calibrator.find_optimal(
        mode=args.mode,
        max_risk=args.max_risk,
        min_speedup=args.min_speedup,
        target_speedup=args.target_speedup,
    )
    
    # Save config
    config_path = Path(args.output)
    with open(config_path, "w") as f:
        f.write(optimal.to_json())
    print(f"\n[SAVED] Config saved to: {config_path}")
    
    # Print env vars for runtime
    print("\n--- Environment Variables for Runtime ---")
    for k, v in optimal.to_env_vars().items():
        print(f"  export {k}={v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
