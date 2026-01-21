# Changelog

## V6.1 (2025-01)

### Major Innovations

- **Confidence Accumulation**: Cross-layer confidence aggregation instead of single-point decisions
  - Bayesian-inspired update: `C_i = C_{i-1} × γ + conf_i × w_i`
  - Reduces noise sensitivity, requires consistent confidence across layers
  - Configurable decay factor (`SEDAC_CONFIDENCE_DECAY`) and per-layer weights (`SEDAC_LAYER_WEIGHTS`)

- **Soft Exit**: Gradual MLP computation reduction instead of hard skipping
  - Smooth transition based on accumulated confidence
  - Avoids sharp quality cliffs at decision boundaries
  - Configurable via `SEDAC_SOFT_EXIT`

- **Rust Core** (`sedac-core`): High-performance implementation
  - Parallel batch evaluation using Rayon
  - Lock-free atomic threshold updates
  - Python fallback for compatibility

### Technical Details

- Exit condition: `accumulated_confidence >= target_exit_rate`
- Soft exit ratio: `tanh(2C - 1) × 0.5 + 0.5`
- EMA threshold smoothing with configurable alpha

### New Files

- `sedac-core/` - Rust crate with PyO3 bindings
- `patch_vllm_surgical_v61.py` - V6.1 vLLM patch
- `docs/SEDAC_V6.1_DESIGN.md` - Design document

### Environment Variables (New)

| Variable | Default | Description |
|----------|---------|-------------|
| `SEDAC_CONFIDENCE_DECAY` | `0.9` | Decay factor for confidence accumulation |
| `SEDAC_SOFT_EXIT` | `1` | Enable soft exit mode |
| `SEDAC_LAYER_WEIGHTS` | `0.3,0.4,0.3` | Per-layer confidence weights |

---

## V6.0 (2024-01)

### Major Changes

- **Multi-layer Cascade Exit**: Replaced single-layer exit with cascade architecture
  - Probes at layers 7, 14, 21 for progressive confidence assessment
  - Token-level early exit decisions at each checkpoint

- **Adaptive Threshold Calibration**: Runtime threshold optimization
  - Collects risk distribution during warmup phase
  - Automatically calibrates thresholds based on target exit rates
  - Configurable via `SEDAC_ADAPTIVE` and `SEDAC_EXIT_RATES`

- **Per-layer Entropy Training**: Fixed data collection methodology
  - Each layer now trained on its own exit entropy (not shared final-layer entropy)
  - Improved probe prediction accuracy and correlation

### Performance

- Probe inference: ~5-8M tokens/sec
- Theoretical speedup: 1.08x-1.25x (task-dependent)
- Memory overhead: <1MB per probe

### Breaking Changes

- Environment variables renamed for V6 compatibility
- Probe file format: `sedac_probe_layer{N}.pth` (layer-specific)
- Removed single-layer mode (`SEDAC_LAYER` deprecated)

### Files

- `patch_vllm_surgical.py` - V6 vLLM patch
- `collect_multilayer_data.py` - Multi-layer data collection
- `train_multilayer_probes.py` - Batch probe training
- `test_v6_local.py` - Local validation suite

---

## V5.0 (2023)

- Initial release with single-layer early exit
- Fixed exit layer configuration
- Basic threshold calibration
