# SEDAC V7.0 Speedup Test Report

**Test Time**: 2026-01-21 21:22:43

## Configuration

- **Data**: `sedac_data_v7_full/`
- **Layers**: 36 (full model)
- **Samples**: 1000
- **Model**: Qwen/Qwen2.5-3B-Instruct

## Results

| tau | K | Avg Exit Layer | Speedup | Exit Rate | High Risk |
|-----|---|----------------|---------|-----------|----------|
| 0.99 | 3 | 32.5 | **1.11x** | 22.5% | 9.00% |
| 0.98 | 3 | 25.1 | **1.44x** | 67.3% | 22.40% |
| 0.98 | 2 | 20.1 | **1.79x** | 79.1% | 22.40% |
| 0.97 | 2 | 9.0 | **3.98x** | 98.5% | 23.80% |
| 0.95 | 2 | 4.1 | **8.83x** | 100.0% | 23.80% |
| 0.9 | 2 | 4.0 | **8.93x** | 100.0% | 23.80% |

## Stability Analysis

- Layers with stability > 0.9: **35/35**
- Average inter-layer stability: **0.9680**

## Recommended Configurations

| Use Case | tau | K | Expected Speedup |
|----------|-----|---|------------------|
| Conservative (quality-first) | 0.98-0.99 | 3 | 1.5-2x |
| Balanced | 0.97-0.98 | 2 | 2-3x |
| Aggressive (speed-first) | 0.90-0.95 | 2 | 5-7x |
