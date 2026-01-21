"""
SEDAC Universal Per-Layer Monitor
==================================

Architecture-agnostic full-layer entropy monitoring.

Design Principles:
1. Zero Model Assumption - No dependency on specific architectures (Qwen, Llama, Mistral, GPT, etc.)
2. Zero Training Dependency - No probe training required, uses hidden state statistics
3. Single Parameter Tuning - Only one threshold tau, avoiding multi-threshold coupling
4. Hook Mechanism - Auto-detect layer structure via PyTorch native hooks

Core Metric: Semantic Stability = Cosine similarity between adjacent layer hidden states
- High stability -> Semantics converged -> Can exit
- Low stability -> Semantics changing -> Continue computation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyExitException(Exception):
    """Early exit signal with cached output."""
    def __init__(self, hidden: torch.Tensor, layer_idx: int):
        self.hidden = hidden
        self.layer_idx = layer_idx


@dataclass
class UniversalConfig:
    """
    Universal Configuration - Architecture-agnostic parameters.
    
    Args:
        tau: Stability threshold, higher = more conservative
        consecutive_k: Required consecutive stable layers
        min_layer_ratio: Minimum exit layer ratio (relative to total layers)
        warmup_ratio: Warmup layer ratio
    """
    tau: float = 0.92
    consecutive_k: int = 3
    min_layer_ratio: float = 0.2
    warmup_ratio: float = 0.1


def auto_detect_layers(model: nn.Module) -> List[nn.Module]:
    """
    Auto-detect transformer layers in a model.
    
    Supported patterns:
    - model.layers (Qwen, Llama)
    - model.model.layers (HuggingFace wrapper)
    - model.transformer.h (GPT-2, GPT-Neo)
    - model.encoder.layer (BERT)
    - model.decoder.layers (T5 decoder)
    
    Falls back to iterating all Sequential modules if auto-detection fails.
    """
    patterns = [
        ("model", "layers"),
        ("layers",),
        ("transformer", "h"),
        ("encoder", "layer"),
        ("decoder", "layers"),
        ("model", "decoder", "layers"),
    ]
    
    for pattern in patterns:
        obj = model
        for attr in pattern:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, '__len__') and len(obj) > 0:
            return list(obj)
    
    candidates = []
    for name, module in model.named_modules():
        if re.search(r'layer|block|h\.\d+', name, re.IGNORECASE):
            candidates.append(module)
    
    if candidates:
        return candidates
    
    raise ValueError(
        "Cannot auto-detect transformer layers. "
        "Please provide layers explicitly via `layers` parameter."
    )


class UniversalStabilityMonitor:
    """
    Universal Stability Monitor.
    
    Monitors semantic stability by computing cosine similarity between adjacent layers.
    When K consecutive layers exceed stability threshold tau, semantics are considered
    converged and early exit is triggered.
    
    Example:
        ```python
        # Any model
        model = AutoModel.from_pretrained("any-model")
        monitor = UniversalStabilityMonitor()
        
        # Auto-detect layers
        layers = auto_detect_layers(model)
        
        for i, layer in enumerate(layers):
            hidden = layer(hidden)
            should_exit = monitor.step(hidden, i, len(layers))
            if should_exit:
                break
        ```
    """
    
    def __init__(self, config: Optional[UniversalConfig] = None):
        self.config = config or UniversalConfig()
        self.reset()
    
    def reset(self) -> None:
        """Reset state for new sequence."""
        self._prev_hidden: Optional[torch.Tensor] = None
        self._consecutive_stable: int = 0
        self._history: List[Dict[str, float]] = []
    
    def compute_stability(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> float:
        """
        Compute stability between two layers (architecture-agnostic).
        
        Uses the last token's hidden state to compute cosine similarity.
        """
        if current.dim() == 3:
            curr = current[:, -1, :].flatten()
            prev = previous[:, -1, :].flatten()
        elif current.dim() == 2:
            curr = current[-1, :] if current.shape[0] > 1 else current.flatten()
            prev = previous[-1, :] if previous.shape[0] > 1 else previous.flatten()
        else:
            curr = current.flatten()
            prev = previous.flatten()
        
        cos_sim = F.cosine_similarity(
            curr.unsqueeze(0).float(),
            prev.unsqueeze(0).float(),
            dim=-1
        ).item()
        
        return (cos_sim + 1) / 2
    
    def step(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        total_layers: int,
    ) -> bool:
        """
        Process a single layer and return whether to exit.
        
        Args:
            hidden: Current layer output
            layer_idx: Current layer index (0-indexed)
            total_layers: Total number of layers
        
        Returns:
            should_exit: Whether to trigger early exit
        """
        min_layer = int(total_layers * self.config.min_layer_ratio)
        warmup_layer = int(total_layers * self.config.warmup_ratio)
        
        record = {
            "layer": layer_idx,
            "stability": 0.0,
            "consecutive": self._consecutive_stable,
        }
        
        if layer_idx < warmup_layer or self._prev_hidden is None:
            self._prev_hidden = hidden.detach().clone()
            self._history.append(record)
            return False
        
        stability = self.compute_stability(hidden, self._prev_hidden)
        record["stability"] = stability
        
        if stability >= self.config.tau:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0
        
        record["consecutive"] = self._consecutive_stable
        self._history.append(record)
        
        self._prev_hidden = hidden.detach().clone()
        
        should_exit = (
            layer_idx >= min_layer and
            self._consecutive_stable >= self.config.consecutive_k
        )
        
        return should_exit
    
    def get_history(self) -> List[Dict[str, float]]:
        """Get stability history."""
        return self._history.copy()


class UniversalHook:
    """
    Universal Hook Mechanism - Auto-inject into any Transformer.
    
    Auto-detects model layer structure and registers forward hooks for exit monitoring.
    
    Example:
        ```python
        model = AutoModelForCausalLM.from_pretrained("any-model")
        
        with UniversalHook(model, config=UniversalConfig(tau=0.92)) as hook:
            try:
                output = model(input_ids)
            except EarlyExitException as e:
                # Early exit
                output = project_to_vocab(e.hidden)
        ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[UniversalConfig] = None,
        layers: Optional[List[nn.Module]] = None,
    ):
        self.model = model
        self.config = config or UniversalConfig()
        self.monitor = UniversalStabilityMonitor(config)
        
        self._layers = layers or auto_detect_layers(model)
        self._num_layers = len(self._layers)
        self._hooks: List[Any] = []
        self._enabled = True
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks on all layers."""
        for i, layer in enumerate(self._layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)
    
    def _make_hook(self, layer_idx: int) -> Callable:
        """Create hook function for a layer."""
        def hook_fn(module, input, output):
            if not self._enabled:
                return output
            
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            should_exit = self.monitor.step(hidden, layer_idx, self._num_layers)
            
            if should_exit:
                raise EarlyExitException(hidden, layer_idx)
            
            return output
        
        return hook_fn
    
    def reset(self) -> None:
        """Reset monitor for new sequence."""
        self.monitor.reset()
    
    def enable(self) -> None:
        """Enable early exit."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable early exit."""
        self._enabled = False
    
    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        if exc_type is EarlyExitException:
            return False
        return False
    
    @property
    def num_layers(self) -> int:
        return self._num_layers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        history = self.monitor.get_history()
        if not history:
            return {}
        
        stabilities = [h["stability"] for h in history if h["stability"] > 0]
        return {
            "num_layers": self._num_layers,
            "processed_layers": len(history),
            "avg_stability": sum(stabilities) / len(stabilities) if stabilities else 0,
            "history": history,
        }


def create_universal_accelerator(
    model: nn.Module,
    tau: float = 0.92,
    consecutive_k: int = 3,
    min_layer_ratio: float = 0.2,
) -> UniversalHook:
    """
    Convenience function to create a universal accelerator.
    
    Args:
        model: Any Transformer model
        tau: Stability threshold
        consecutive_k: Consecutive stable layers required
        min_layer_ratio: Minimum exit layer ratio
    
    Returns:
        UniversalHook instance
    
    Example:
        ```python
        accelerator = create_universal_accelerator(model, tau=0.9)
        
        with accelerator:
            try:
                out = model(inputs)
            except EarlyExitException as e:
                out = e.hidden
        ```
    """
    config = UniversalConfig(
        tau=tau,
        consecutive_k=consecutive_k,
        min_layer_ratio=min_layer_ratio,
    )
    return UniversalHook(model, config)
