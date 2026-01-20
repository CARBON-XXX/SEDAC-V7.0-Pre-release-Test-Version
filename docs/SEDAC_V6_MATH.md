# SEDAC V6.0 Mathematical Formulation & Derivations

## 1. Low-Rank Entropy Probe (LREProbe)

The core risk estimator is a lightweight MLP attached to the hidden states $h_l \in \mathbb{R}^{B \times S \times D}$ of layer $l$.

$$
r(h) = \text{Softplus}(W_2 \cdot \text{LayerNorm}(W_1 h))
$$

Where:
- $W_1 \in \mathbb{R}^{D \times R}$ projects the high-dimensional hidden state to a low-rank subspace $R \ll D$ (typically $R=64$).
- $W_2 \in \mathbb{R}^{R \times 1}$ maps the normalized features to a scalar logit.
- $\text{Softplus}(x) = \log(1 + e^x)$ ensures non-negative risk scores.

**Key Assumption**: The "uncertainty" or "difficulty" of a token is linearly separable in a low-rank subspace of the hidden states after LayerNorm.

## 2. Cascade Exit Decision (Sequence-Level Latch)

In SEDAC V6, we perform early exit checks at multiple checkpoints $L_{check} = \{7, 14, 21\}$.
Unlike token-level masking (which is complex to implement safely with KV-cache), we use a **Sequence-Level Latch** mechanism.

For a sequence of length $S$ at layer $l \in L_{check}$, the exit condition is:

$$
E_l = \mathbb{I}\left(\max_{t=1}^S r_l(h_{l,t}) < \tau_l\right)
$$

If $E_l = 1$:
- The request is marked as "exited".
- For all subsequent layers $k > l$, the MLP block is skipped: $h_{k} = h_{k-1}$ (effectively).
- Attention blocks are **still executed** to update the KV-cache, ensuring future tokens can attend to this position correctly.

**Why Max?**
Using $\max$ is a conservative safety guarantee. If *any* token in the sequence has high risk (high uncertainty), we do *not* exit, because that token might need further processing (depth) to be resolved correctly. We only exit if *all* tokens are "safe" (low risk).

## 3. Adaptive Threshold Calibration

To maintain a target exit rate $\rho_{target}$ (e.g., 0.5) under shifting data distributions, we adjust $\tau_l$ dynamically.

Let $R_l = \{r_{l,1}, r_{l,2}, \dots, r_{l, N}\}$ be a buffer of observed max-risk scores at layer $l$.
The raw target threshold $\hat{\tau}$ is the $q$-th quantile of $R_l$, where $q = \rho_{target}$.

$$
\hat{\tau}_l = \text{Quantile}(R_l, \rho_{target})
$$

### Alpha Smoothing ($\alpha$)

To prevent threshold oscillation due to batch noise, we apply Exponential Moving Average (EMA) smoothing:

$$
\tau_{l, t+1} = (1 - \alpha) \cdot \tau_{l, t} + \alpha \cdot \hat{\tau}_{l, batch}
$$

Where:
- $\alpha \in [0, 1]$ is the smoothing factor (`SEDAC_CALIBRATION_ALPHA`).
- $\alpha \to 1$: Fast adaptation, high variance.
- $\alpha \to 0$: Slow adaptation, stable thresholds.
- Default $\alpha=1.0$ (instant update) for initial calibration, but lower values recommended for continuous online adaptation.

## 4. Complexity Analysis

- **Probe Cost**: $O(S \cdot D \cdot R)$ per checkpoint.
- **MLP Savings**: $O(S \cdot D^2)$ per skipped layer.
- **Break-even Condition**:
  $$
  N_{skipped} \cdot (S \cdot D^2) > |L_{check}| \cdot (S \cdot D \cdot R)
  $$
  Since $R \approx 64$ and $D \approx 2048$, $D/R \approx 32$. One skipped MLP layer pays for ~32 probe executions.