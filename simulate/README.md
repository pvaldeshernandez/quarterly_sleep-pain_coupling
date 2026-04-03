# Synthetic Data Generator

This directory contains the parametric data generator that produces synthetic
datasets mirroring the real UPLOAD2 knee-pain cohort.  The synthetic data
lets the full Python analysis pipeline run end-to-end without access to
restricted participant data.

## Quick start

```bash
conda activate sleep-pain-coupling
python simulate/generate_synthetic_data.py --seed 42
```

## What the generator does

The script simulates data from the same Vector Autoregressive model with
exogenous input (VARX(1)) that is fitted in the analysis.  Ground-truth
parameters are hard-coded from the manuscript tables, so fitting the model
to the synthetic data should recover those parameters (within sampling
noise).

### Step-by-step

1. **Demographics** (N = 229).  Age is drawn from a truncated normal
   (mean 58.1, SD 8.2, range 31--79).  Sex is Bernoulli (64.6% female).
   Race is multinomial (39.7% Black, 29.3% White, 24.0% Hispanic,
   7.0% Other).

2. **Timepoint counts**.  Each person gets 2--10 quarterly visits drawn
   from a left-skewed discrete distribution calibrated to match the
   empirical median of 9 and a total of ~1818 usable transitions.

3. **Random effects**.  For each person, two independent Gaussian random
   effects are drawn:
   - u_sp ~ N(0, 0.115²) -- sleep-to-pain coupling heterogeneity
   - u_ps ~ N(0, 0.362²) -- pain-to-sleep coupling heterogeneity

4. **VARX(1) forward simulation**.  For each person, the bivariate time
   series is iterated forward from a random initial condition:

   ```
   pain(t)  = mu_p + phi_p * pain(t-1) + lambda_sp_i * sleep(t-1)
            + delta_p * K(t-1) + omega_sp * sleep(t-1) * K(t-1) + eps_p

   sleep(t) = mu_s + lambda_ps_i * pain(t-1) + phi_s * sleep(t-1)
            + delta_s * K(t-1) + omega_ps * pain(t-1) * K(t-1) + eps_s
   ```

   where K is the contrast factor (exogenous, ~N(0,1)), and the person-
   specific coupling includes age/sex moderation plus the random effect.

   **Innovations** (eps_p, eps_s) are correlated via the Cholesky trick:
   two independent standard normals are transformed to achieve the target
   correlation (rho = -0.155) without constructing a covariance matrix.

5. **Within-person centering**.  After generating the raw series, each
   person's pain, sleep, and contrast values are mean-subtracted.  This
   matches the preprocessing applied to the real data.

6. **fMRI subsample** (N = 174).  Twelve BOLD ROI z-scores are generated.
   For the two ROIs with significant sleep-to-pain moderation (left NAcc,
   right dACC/MCC), values are constructed with an embedded correlation
   with the u_sp random effect so the moderation signal is recoverable.
   Five VBM gray matter volumes are generated with a weak directional
   bias matching the observed sign concordance (5/5 negative gamma_ps).

## Output files

| File | Description |
|------|-------------|
| `data/synthetic/quarterly_data_long.csv` | Longitudinal data: subject_id, quarter, pain_severity, contrast_factor, sleep_quality |
| `data/synthetic/participants_wideformat.csv` | Demographics: subject_id, age, sex, race |
| `data/synthetic/roi_values.csv` | fMRI BOLD ROI values (long format, z-scored) |
| `data/synthetic/vbm_volumes.csv` | VBM gray matter volumes (long format) |
| `data/synthetic/ground_truth.json` | All parameters used for generation |

## Ground-truth parameters

See `data/synthetic/ground_truth.json` for the complete parameter set.
Key values (from manuscript Table 4):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| lambda_sp | -0.021 | Sleep-to-pain coupling (weak) |
| lambda_ps | -0.140 | Pain-to-sleep coupling (dominant) |
| tau_sp | 0.115 | Random-effect SD, sleep-to-pain |
| tau_ps | 0.362 | Random-effect SD, pain-to-sleep |
| sigma_p | 0.437 | Pain innovation SD |
| sigma_s | 0.661 | Sleep innovation SD |
| rho | -0.155 | Innovation cross-correlation |

## Reproducibility

The generator uses `numpy.random.default_rng(seed)` throughout.  The
default seed is 42.  Passing the same seed always produces identical
output.
