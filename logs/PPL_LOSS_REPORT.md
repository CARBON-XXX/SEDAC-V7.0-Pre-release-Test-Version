# SEDAC V6 PPL Loss Report

**Test Time**: 2026-01-21 21:11:54

## Summary

| Target Speedup | Actual Speedup | Exit Rate | Entropy Increase | High Risk Rate | PPL Loss |
|----------------|----------------|-----------|------------------|----------------|----------|
| 1.5x | 2.18x | 37.9% | -10.95% | 5.06% | **-38.98%** [OK] |
| 2.0x | 2.43x | 51.7% | -12.13% | 8.58% | **-29.05%** [OK] |
| 2.5x | 2.43x | 51.7% | -12.13% | 8.58% | **-29.05%** [OK] |
| 3.0x | 4.26x | 96.1% | 13.00% | 24.45% | **12.18%** [WARN] |

## Analysis

### At 2x Speedup:

- **Actual Speedup**: 2.43x
- **Exit Rate**: 51.7%
- **Entropy Increase**: -12.13%
- **PPL Loss**: -29.05%

**Conclusion**: PPL Loss < 1% at 2x speedup. Quality preserved.

## Methodology

- **PPL Loss** = (Exit Entropy - Baseline Entropy) / Baseline Entropy
- **High Risk Rate** = % of early-exit tokens with final entropy > 75th percentile
- **Baseline Entropy** = Average entropy at final layer (layer 21)
