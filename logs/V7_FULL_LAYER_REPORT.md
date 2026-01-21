# SEDAC V7 Full-Layer Training Report

**Test Time**: 2026-01-21 21:10

## Data Collection

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Layers**: 36 (0-35) - ALL LAYERS
- **Samples**: 1340 tokens
- **Output**: `sedac_data_v7_full/`

## Layer Stability Analysis

全部 35 个相邻层对的稳定性 > 0.9，证明语义在深层逐渐收敛。

| Layer Range | Avg Stability |
|-------------|---------------|
| 1-10 | 0.96 |
| 11-20 | 0.95 |
| 21-30 | 0.97 |
| 31-35 | 0.97 |

## V7 Speedup Results

| Config (tau, K) | Avg Exit Layer | Speedup | Exit Rate |
|-----------------|----------------|---------|-----------|
| tau=0.98, K=2 | 19.9 | **1.81x** | 79.4% |
| tau=0.98, K=3 | 23.2 | **1.55x** | 72.4% |
| tau=0.95, K=2 | 5.2 | **6.88x** | 100% |
| tau=0.95, K=3 | 5.4 | **6.64x** | 100% |
| tau=0.90, K=2 | 5.0 | **7.18x** | 100% |

## Key Findings

1. **V7 works with full-layer data**: With all 36 layers, V7 can detect semantic stability
2. **High speedup potential**: tau=0.95 achieves 6-7x speedup
3. **Conservative option**: tau=0.98, K=3 gives 1.55x with 72% exit rate

## Comparison: V6 vs V7

| Metric | V6 (Probe) | V7 (Universal) |
|--------|------------|----------------|
| Training Required | Yes (6 probes) | **No** |
| Architecture Dep. | Model-specific | **Universal** |
| Speedup (conservative) | 2.21x | 1.55x-1.81x |
| Speedup (aggressive) | - | 6-7x |
| Parameters | 6 thresholds | 2 (tau, K) |
