"""
Pure Python fallback implementation for sedac_core.

This provides the same API as the Rust extension but with pure Python/NumPy.
Used when the Rust extension is not available.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import threading


@dataclass
class LayerConfig:
    """Configuration for a checkpoint layer."""
    layer_idx: int
    target_exit_rate: float
    initial_threshold: float
    confidence_weight: float


@dataclass
class ExitDecision:
    """Result of cascade exit evaluation."""
    should_exit: bool
    exit_layer: int
    accumulated_confidence: float
    soft_exit_ratio: float


class CascadeController:
    """
    Pure Python implementation of CascadeController.
    
    Provides confidence accumulation and soft exit mechanisms for V6.1.
    """

    def __init__(
        self,
        layer_configs: List[LayerConfig],
        alpha: float = 0.1,
        confidence_decay: float = 0.9,
        soft_exit_enabled: bool = True,
        calibration_steps: int = 50,
    ):
        self.layer_configs = layer_configs
        self.alpha = alpha
        self.confidence_decay = confidence_decay
        self.soft_exit_enabled = soft_exit_enabled
        self.calibration_steps = calibration_steps

        self.thresholds = np.array([cfg.initial_threshold for cfg in layer_configs], dtype=np.float32)
        self.calibration_samples = [[] for _ in layer_configs]
        self.is_calibrated_flag = False
        self.total_calls = 0
        self.total_exits = [0] * len(layer_configs)
        self._lock = threading.Lock()

    def evaluate_batch(self, risk_scores: np.ndarray) -> List[ExitDecision]:
        """
        Evaluate cascade exit decision for a batch of tokens.

        Args:
            risk_scores: 2D array [num_layers, batch_size] of risk scores

        Returns:
            List of ExitDecision for each token
        """
        num_layers, batch_size = risk_scores.shape
        self.total_calls += batch_size

        decisions = []
        for token_idx in range(batch_size):
            accumulated_conf = 0.0
            exit_layer = -1
            should_exit = False

            for layer_i, cfg in enumerate(self.layer_configs):
                risk = risk_scores[layer_i, token_idx]
                threshold = self.thresholds[layer_i]

                # Confidence = how much risk is below threshold
                if threshold > 0:
                    layer_confidence = max(0.0, min(1.0, (threshold - risk) / threshold))
                else:
                    layer_confidence = 0.0

                # Bayesian-style accumulation with decay
                accumulated_conf = accumulated_conf * self.confidence_decay + layer_confidence * cfg.confidence_weight

                # Exit condition
                if accumulated_conf >= cfg.target_exit_rate and not should_exit:
                    should_exit = True
                    exit_layer = cfg.layer_idx

            # Soft exit ratio
            if self.soft_exit_enabled and should_exit:
                soft_exit_ratio = np.tanh(accumulated_conf * 2.0 - 1.0) * 0.5 + 0.5
            elif should_exit:
                soft_exit_ratio = 1.0
            else:
                soft_exit_ratio = 0.0

            decisions.append(ExitDecision(
                should_exit=should_exit,
                exit_layer=exit_layer,
                accumulated_confidence=accumulated_conf,
                soft_exit_ratio=soft_exit_ratio,
            ))

            # Update exit counters
            if should_exit and exit_layer >= 0:
                for i, cfg in enumerate(self.layer_configs):
                    if cfg.layer_idx == exit_layer:
                        self.total_exits[i] += 1
                        break

        return decisions

    def update_threshold(self, layer_idx: int, risk_samples: np.ndarray) -> float:
        """Update threshold with new risk samples using EMA."""
        with self._lock:
            if not self.is_calibrated_flag:
                # Warmup phase
                self.calibration_samples[layer_idx].extend(risk_samples.tolist())

                all_ready = all(
                    len(s) >= self.calibration_steps for s in self.calibration_samples
                )

                if all_ready:
                    # Initial calibration
                    for i, cfg in enumerate(self.layer_configs):
                        sorted_samples = np.sort(self.calibration_samples[i])
                        q_idx = int(len(sorted_samples) * cfg.target_exit_rate)
                        q_idx = min(q_idx, len(sorted_samples) - 1)
                        self.thresholds[i] = sorted_samples[q_idx]

                    self.is_calibrated_flag = True

                return self.thresholds[layer_idx]

            # Online calibration with EMA
            if len(risk_samples) > 0:
                sorted_samples = np.sort(risk_samples)
                target_rate = self.layer_configs[layer_idx].target_exit_rate
                q_idx = int(len(sorted_samples) * target_rate)
                q_idx = min(q_idx, len(sorted_samples) - 1)

                batch_threshold = sorted_samples[q_idx]
                old_threshold = self.thresholds[layer_idx]
                new_threshold = self.alpha * batch_threshold + (1 - self.alpha) * old_threshold
                self.thresholds[layer_idx] = new_threshold

                return new_threshold

            return self.thresholds[layer_idx]

    def get_threshold(self, layer_idx: int) -> float:
        return self.thresholds[layer_idx] if layer_idx < len(self.thresholds) else 0.0

    def get_all_thresholds(self) -> np.ndarray:
        return self.thresholds.copy()

    def get_stats(self) -> Tuple[int, List[int]]:
        return self.total_calls, self.total_exits.copy()

    def is_calibrated(self) -> bool:
        return self.is_calibrated_flag

    def reset_calibration(self) -> None:
        with self._lock:
            self.is_calibrated_flag = False
            self.calibration_samples = [[] for _ in self.layer_configs]


def compute_quantile(data: np.ndarray, quantile: float) -> float:
    """Compute quantile of a 1D array."""
    if len(data) == 0:
        return 0.0
    return float(np.percentile(data, quantile * 100))


def batch_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax for a batch of logits."""
    max_vals = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_vals)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute entropy from probability distributions."""
    # Avoid log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    return entropy
