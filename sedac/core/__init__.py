"""
SEDAC Core Module
=================

Core components for cascade early exit:
- CascadeController: Multi-layer exit decision with confidence accumulation
- ProbeManager: Efficient probe inference with JIT and CUDA graph support
- ExitStrategy: Configurable exit strategies (hard, soft, adaptive)
- UniversalMonitor: Architecture-agnostic per-layer entropy monitoring (V7)
"""

from sedac.core.cascade_controller import (
    CascadeController,
    CascadeConfig,
    LayerConfig,
)
from sedac.core.probe_inference import LREProbe, ProbeManager
from sedac.core.exit_strategy import (
    ExitStrategy,
    HardExit,
    SoftExit,
    AdaptiveSoftExit,
    ExitDecision,
)
from sedac.core.universal_monitor import (
    UniversalStabilityMonitor,
    UniversalHook,
    UniversalConfig,
    EarlyExitException,
    auto_detect_layers,
    create_universal_accelerator,
)

__all__ = [
    # V6 components (model-specific probes)
    "CascadeController",
    "CascadeConfig",
    "LayerConfig",
    "LREProbe",
    "ProbeManager",
    "ExitStrategy",
    "HardExit",
    "SoftExit",
    "AdaptiveSoftExit",
    "ExitDecision",
    # V7 components (architecture-agnostic)
    "UniversalStabilityMonitor",
    "UniversalHook",
    "UniversalConfig",
    "EarlyExitException",
    "auto_detect_layers",
    "create_universal_accelerator",
]
