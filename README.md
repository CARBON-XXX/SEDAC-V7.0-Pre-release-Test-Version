# SEDAC V7.0: Universal Semantic Entropy Dynamic Acceleration Core

**SEDAC** (Semantic Entropy Dynamic Acceleration Core) is a high-performance framework for accelerating LLM inference by dynamically skipping redundant layers for "easy" tokens.

> **Latest Update (V7.0)**: Introduces **Architecture-Agnostic Universal Monitoring**, eliminating the need for model-specific training.

---

## ⚡ Performance Benchmark

| Metric | **V6.1 (Probe-Based)** | **V7.0 (Universal)** |
|--------|------------------------|----------------------|
| **Mechanism** | Trained LRE Probes | Cosine Stability |
| **Setup** | Requires Training (Hours) | **Zero-Shot (Instant)** |
| **Architecture** | Model-Specific | **Universal** |
| **Speedup (Balanced)** | **~2.21x** | ~1.81x |
| **Speedup (Aggressive)** | ~3.0x | **~8.83x** |
| **Robustness** | Medium (Single Check) | **High (Consecutive K)** |

### ❓ Why is V6 sometimes faster than V7?

You may notice V6 (2.21x) outperforming V7 (1.81x) in "Balanced" configurations. This is by design:

1.  **Exit Condition Latency**:
    - **V6** exits *immediately* when a probe predicts low entropy. It is "eager."
    - **V7** requires **K consecutive layers** (default K=2 or 3) to be stable. This "verification window" ensures higher quality but delays the exit by K-1 layers.

2.  **Checkpointing**:
    - **V6** typically checks only specific layers (e.g., 6, 12, 18).
    - **V7** checks **every layer**, allowing for finer-grained exits (e.g., layer 9, 23) but incurring slight monitoring overhead at every step.

**Conclusion**: Use **V6** for maximum speed in specific checkpoints if you have trained probes. Use **V7** for universal compatibility and robustness without training.

---

## 🛠️ Project Structure

```
SEDACV5.0 FAST/
├── sedac/
│   └── core/
│       ├── universal_monitor.py   # [V7] Universal Stability Monitor
│       ├── probe_inference.py     # [V6] LRE Probe Inference
│       └── __init__.py
├── sedac-core/                    # [Rust] High-Performance Core
│   └── src/
│       └── lib.rs                 # <--- Rust Implementation (Bayesian, EMA)
├── sedac_calibrate_auto.py        # [Tool] Auto-Calibration System
├── sedac_collect_data.py          # [Tool] Data Collector
├── sedac_test.py                  # [Tool] Unified Test Framework
├── sedac_config.json              # [Config] Auto-generated Config
└── logs/                          # Test Logs & Reports
```

---

## 🦀 Rust Core Implementation

The high-performance core logic is implemented in Rust to minimize Python GIL overhead.

**Location**: `sedac-core/src/lib.rs`

### Key Features (Rust):
1.  **Async Batch Processing**: Parallel processing of token batches using `rayon`.
2.  **Lock-Free Thresholds**: `AtomicF32` implementation for thread-safe EMA updates.
3.  **Confidence Accumulation**: Bayesian update logic for multi-layer confidence.

```rust
// sedac-core/src/lib.rs

// Bayesian-style accumulation with decay
accumulated_conf = accumulated_conf * self.confidence_decay
    + layer_confidence * cfg.confidence_weight;

// Soft exit ratio calculation
let soft_exit_ratio = if self.soft_exit_enabled && should_exit {
    (accumulated_conf * 2.0 - 1.0).tanh() * 0.5 + 0.5
}
```

---

## 📐 Mathematical Principles

### V7: Semantic Stability (Cosine Similarity)

We measure the convergence of hidden states between layers $l$ and $l-1$:

$$ S_l = \frac{1}{2} \left( \frac{h_l \cdot h_{l-1}}{\|h_l\| \|h_{l-1}\|} + 1 \right) $$

**Exit Condition**:
Exit at layer $L$ if stability exceeds threshold $\tau$ for $K$ consecutive layers:
$$ \forall i \in [L-K+1, L], \quad S_i \geq \tau $$

---

## 🚀 Quick Start

### 1. Auto-Calibration (Recommended)

Let the system find the best configuration for your hardware and quality constraints.

```bash
# Calibrate for balanced performance (max 15% risk)
python sedac_calibrate_auto.py --data-dir sedac_data --mode balanced --max-risk 0.15
```

### 2. Run Unified Test

Validate the configuration with the unified test framework.

```bash
# Run full test suite (V6 + V7)
python sedac_test.py --data-dir sedac_data --full
```

### 3. Usage in Code

```python
from sedac.core.universal_monitor import create_universal_accelerator

# Initialize with auto-calibrated config
accelerator = create_universal_accelerator(model, tau=0.98, consecutive_k=2)

with accelerator:
    output = model(input_ids)
```

---

## License

MIT License.
