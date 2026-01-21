# SEDAC V6.1 Design Document

## Overview

SEDAC V6.1 introduces significant architectural improvements over V6.0, focusing on **innovation and quality** rather than raw speed. The key changes are:

1. **Rust Core** (`sedac-core`): High-performance implementation of cascade logic
2. **Confidence Accumulation**: Cross-layer confidence aggregation instead of single-point decisions
3. **Soft Exit**: Gradual MLP computation reduction instead of hard skipping

---

## Problem Statement

### V6.0 Limitations

1. **Single-Point Decision Risk**: V6.0 makes binary exit decisions at each checkpoint layer independently. A single noisy probe prediction can cause premature exit.

2. **Hard Exit Boundary**: Once exit is triggered, all subsequent MLP computations are skipped (100% skip). This creates a sharp quality cliff.

3. **Python GIL Bottleneck**: Threshold updates and batch evaluations are Python-bound, limiting throughput.

4. **No Cross-Layer Information**: Each layer's decision ignores information from previous layers.

---

## V6.1 Architecture

### 1. Confidence Accumulation

Instead of binary decisions, V6.1 accumulates confidence scores across layers using a Bayesian-inspired update:

```
C_i = C_{i-1} * γ + conf_i * w_i
```

Where:
- `C_i`: Accumulated confidence at layer i
- `γ`: Decay factor (default: 0.9) - prevents over-accumulation
- `conf_i`: Layer-local confidence = (threshold - risk) / threshold
- `w_i`: Layer weight (configurable per layer)

**Exit Condition**: Exit when `C_i >= target_exit_rate_i`

This approach:
- Reduces noise sensitivity (requires consistent confidence across layers)
- Allows earlier exits for truly simple tokens
- Provides graceful degradation for borderline cases

### 2. Soft Exit

Instead of hard MLP skipping, V6.1 supports **gradual computation reduction**:

```
soft_exit_ratio = tanh(2*C - 1) * 0.5 + 0.5
```

This ratio determines what percentage of MLP channels to skip:
- `C = 0.5` → ratio ≈ 0.5 → skip 50% of MLP channels
- `C = 0.8` → ratio ≈ 0.76 → skip 76% of MLP channels
- `C = 1.0` → ratio ≈ 0.88 → skip 88% of MLP channels

**Implementation Options**:

1. **Channel-wise Masking**: Zero out a fraction of FFN intermediate activations
2. **Early Truncation**: Only compute first N% of FFN hidden units
3. **Importance-weighted Skip**: Skip least important channels (requires offline analysis)

### 3. Rust Core

The `sedac-core` crate provides:

```rust
// Core components
pub struct CascadeController {
    layer_configs: Vec<LayerConfig>,
    thresholds: Vec<AtomicF32>,      // Lock-free threshold storage
    alpha: f32,                       // EMA smoothing factor
    confidence_decay: f32,            // γ in accumulation formula
    soft_exit_enabled: bool,
}

// Key operations
impl CascadeController {
    fn evaluate_batch(&self, risk_scores: &Array2<f32>) -> Vec<ExitDecision>;
    fn update_threshold(&self, layer_idx: usize, samples: &[f32]) -> f32;
}
```

**Performance Benefits**:
- Parallel batch evaluation using Rayon
- Lock-free atomic threshold updates
- SIMD-friendly data layouts
- No Python GIL during computation

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEDAC_CONFIDENCE_DECAY` | `0.9` | Decay factor for confidence accumulation |
| `SEDAC_SOFT_EXIT` | `1` | Enable soft exit (gradual MLP reduction) |
| `SEDAC_LAYER_WEIGHTS` | `0.3,0.4,0.3` | Per-layer confidence weights |

### LayerConfig

```python
LayerConfig(
    layer_idx=7,           # Checkpoint layer index
    target_exit_rate=0.2,  # Exit when accumulated confidence >= this
    initial_threshold=0.8, # Initial risk threshold
    confidence_weight=0.3, # Weight in accumulation formula
)
```

---

## Comparison: V6.0 vs V6.1

| Aspect | V6.0 | V6.1 |
|--------|------|------|
| Decision Model | Single-layer binary | Cross-layer accumulated |
| Exit Type | Hard (0% or 100% MLP) | Soft (gradual reduction) |
| Implementation | Pure Python | Rust + Python |
| Noise Resilience | Low | High |
| Quality Cliff | Sharp | Smooth |
| Throughput | Baseline | ~2x (Rust parallelism) |

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [x] Rust crate structure (`sedac-core`)
- [x] `CascadeController` with confidence accumulation
- [x] Python fallback implementation
- [x] PyO3 bindings

### Phase 2: vLLM Integration
- [ ] Update `patch_vllm_surgical.py` for V6.1
- [ ] Soft exit in decoder layer
- [ ] Integrate Rust controller

### Phase 3: Validation
- [ ] Unit tests for Rust core
- [ ] Integration tests with vLLM
- [ ] Quality benchmarks (PPL comparison)
- [ ] Performance benchmarks

---

## Mathematical Justification

### Why Confidence Accumulation?

Consider a token with true difficulty `d ∈ [0, 1]`. Each probe produces a noisy estimate:

```
risk_i = d + ε_i, where ε_i ~ N(0, σ²)
```

**V6.0 (single-layer)**:
- Decision based on single `risk_i < threshold`
- P(wrong decision) = Φ((threshold - d) / σ)

**V6.1 (accumulated)**:
- Aggregates multiple estimates
- By Central Limit Theorem, aggregated estimate has lower variance
- P(wrong decision) decreases with more layers

### Soft Exit Gradient

The soft exit ratio function `f(C) = tanh(2C - 1) * 0.5 + 0.5` provides:

1. **Smooth transition**: No discontinuity at decision boundary
2. **Bounded output**: Always in [0, 1]
3. **Sensitivity control**: Steepest around C = 0.5

---

## Future Work

1. **Learned Weights**: Train confidence weights end-to-end
2. **Adaptive Decay**: Adjust γ based on sequence difficulty
3. **Token-Type Routing**: Different cascade paths for different token types
4. **Speculative Integration**: Combine with speculative decoding
