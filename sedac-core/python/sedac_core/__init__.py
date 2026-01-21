"""
SEDAC Core - High-Performance Rust Backend for V6.1

This module provides Python bindings for the Rust-based cascade controller.

Usage:
    from sedac_core import CascadeController, LayerConfig

    configs = [
        LayerConfig(layer_idx=7, target_exit_rate=0.2, initial_threshold=0.8, confidence_weight=0.3),
        LayerConfig(layer_idx=14, target_exit_rate=0.5, initial_threshold=1.0, confidence_weight=0.4),
        LayerConfig(layer_idx=21, target_exit_rate=0.8, initial_threshold=1.2, confidence_weight=0.3),
    ]

    controller = CascadeController(
        layer_configs=configs,
        alpha=0.1,
        confidence_decay=0.9,
        soft_exit_enabled=True,
        calibration_steps=50,
    )
"""

try:
    from .sedac_core import (
        CascadeController,
        LayerConfig,
        ExitDecision,
        compute_quantile,
        batch_softmax,
        compute_entropy,
    )

    __all__ = [
        "CascadeController",
        "LayerConfig",
        "ExitDecision",
        "compute_quantile",
        "batch_softmax",
        "compute_entropy",
    ]
    RUST_AVAILABLE = True
except ImportError:
    # Fallback to pure Python implementation
    RUST_AVAILABLE = False
    import warnings
    warnings.warn("sedac_core Rust extension not available, using pure Python fallback")

    from .fallback import (
        CascadeController,
        LayerConfig,
        ExitDecision,
        compute_quantile,
        batch_softmax,
        compute_entropy,
    )

    __all__ = [
        "CascadeController",
        "LayerConfig",
        "ExitDecision",
        "compute_quantile",
        "batch_softmax",
        "compute_entropy",
        "RUST_AVAILABLE",
    ]
