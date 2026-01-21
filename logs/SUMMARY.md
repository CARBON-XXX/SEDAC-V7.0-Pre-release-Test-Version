# SEDAC Test Logs Summary

**Date**: 2026-01-21

## Test Results Overview

### V6.1 Probe-Based Results

| Metric | Value |
|--------|-------|
| **Speedup** | 2.21x |
| **Exit Rate** | 75.6% |
| **PPL Loss @ 2x** | -29.05% (improved) |

### V7.0 Universal Results

| Config | Speedup | Exit Rate | High Risk |
|--------|---------|-----------|-----------|
| tau=0.99, K=3 | 1.11x | 22.5% | 9.0% |
| tau=0.98, K=2 | **1.79x** | 79.1% | 22.4% |
| tau=0.97, K=2 | **3.98x** | 98.5% | 23.8% |
| tau=0.95, K=2 | **8.83x** | 100% | 23.8% |

---

## Log Files

| File | Description |
|------|-------------|
| `01_data_check.log` | Data collection verification |
| `02_train_probes.log` | V6 probe training output |
| `03_test_local.log` | Local validation test |
| `speedup_test.log` | V6 speedup benchmark |
| `ppl_loss_test.log` | V6 PPL loss analysis |
| `v7_speedup_test.log` | V7 full-layer speedup test |

## Reports

| File | Description |
|------|-------------|
| `TEST_REPORT.md` | End-to-end test summary |
| `SPEEDUP_REPORT.md` | V6 speedup results |
| `PPL_LOSS_REPORT.md` | V6 quality analysis |
| `V7_SPEEDUP_REPORT.md` | V7 speedup results |
| `V7_FULL_LAYER_REPORT.md` | V7 full-layer analysis |

---

## Key Findings

1. **V6 PPL Loss is Negative**: Probes select high-confidence tokens for early exit, resulting in *better* quality for exited tokens.

2. **V7 Achieves Higher Speedup**: With aggressive settings (tau=0.95), V7 achieves 8.83x speedup vs V6's 2.21x.

3. **Layer Stability**: 35/35 adjacent layer pairs show stability > 0.9, validating the stability-based approach.

4. **Trade-off**: V6 offers better quality control; V7 offers simpler deployment and higher max speedup.

---

## Recommended Configurations

### For Quality-Critical Applications
- Use V6 with tau calibrated for 2x speedup
- PPL Loss: negative (quality improvement)

### For Speed-Critical Applications
- Use V7 with tau=0.97, K=2
- Speedup: 4x with 98.5% exit rate

### For Maximum Speed
- Use V7 with tau=0.95, K=2
- Speedup: 8.83x (100% exit rate)
- Note: Higher risk rate (23.8%)
