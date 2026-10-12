# Latent mean centering: what came out (2026-10-12)

Sandbox: git worktree `UPLOAD2-sandbox-latent`, branch `sandbox/latent-centering`. Nothing in `UPLOAD2/` (manuscript, `results/`, `derivatives/`, numbered steps) was touched.

## What was built

- `lib/coupling_model.py`: new `centering="latent"` option in `fit_bayesian_varx1` (no separate function). The default graph is identical to the committed one: same 21 free variables, log-probability difference exactly 0 at three points (`check_default_graph.txt`).
- One more option the data required, `first_points=`: each segment's first point is modeled from the person's stationary distribution instead of being conditioned on. This is the handoff's "optional refinement" (section 5). It turned out to be necessary.
- `codes/python/extensions/latent_centering.py` (simulate, fit, report); all outputs are in this folder.

## Did latent centering recover the truth in simulation?

**Only with the stationary first point.** Three datasets were simulated on the real panel (true phi_p 0.27, phi_s 0.15, lambda_sp −0.03, lambda_ps −0.15). Means over the three replicates:

| | True | Observed centering (published method) | Latent, first point conditioned | Latent, first point stationary |
|---|---|---|---|---|
| phi_p | 0.270 | 0.115 | 0.565 | **0.281** |
| phi_s | 0.150 | −0.001 | 0.186 | **0.137** |
| lambda_sp | −0.030 | −0.028 | −0.067 | **−0.036** |
| lambda_ps | −0.150 | −0.142 | −0.250 | **−0.154** |

- Observed centering reproduces the published pattern: memories biased toward zero, couplings close to the truth.
- Latent centering as sketched in the handoff (first point conditioned) **fails**. Its autoregression is biased upward and the fits do not converge (sim 1: R-hat 1.71, 2,328 divergences). This is the classic initial-conditions problem of random-intercept dynamic panels: with about nine quarters per person, the model cannot tell high persistence apart from widely spread usual levels. By the handoff's stop rule, this version was not used on the real data.
- With the stationary first point, the truth is recovered. Across the 3 replicates × 4 key parameters, the 95% CrI covered the truth 11 times out of 12; the miss was phi_p in sim 3 (0.36).

## New memories and couplings (real data, latent + stationary first point)

| | Published | Latent |
|---|---|---|
| phi_p (pain memory) | 0.104 [0.056, 0.153] | **0.293 [0.235, 0.351]** |
| phi_s (sleep memory) | 0.005 [−0.042, 0.051] | **0.148 [0.095, 0.202]** |
| lambda_sp (sleep→pain) | −0.022 [−0.058, 0.014] | −0.029 [−0.066, 0.009] |
| lambda_ps (pain→sleep) | −0.136 [−0.236, −0.039] | **−0.155 [−0.254, −0.054]** |
| standardized lambda_ps | −0.089 | −0.101 |
| standardized lambda_sp | −0.034 | −0.044 |

These are almost exactly the handoff's calibration prediction (0.27, 0.15, −0.032, −0.153).

**Pain→sleep still dominates.** It is credibly negative, P(<0) = 0.998, and slightly larger than published. Sleep→pain remains uncertain (P(<0) = 0.93). The other conclusions do not change. delta_s (−0.056) stays credible. delta_p becomes credible (0.044 [0.013, 0.075]). The Age and Sex moderators are all null. tau_sp, tau_ps and rho_innov barely move. The new between-person correlation of usual pain and usual sleep is rho_alpha = −0.47 [−0.58, −0.34]: people with worse usual pain report worse usual sleep.

## What happens to the step 25 numbers

| | Published | Latent |
|---|---|---|
| Eigenvalues | 0.129, −0.020 (oscillatory) | 0.319, 0.122 (both positive) |
| Reciprocal formula (Eq. S2) | not defined | **58 days** [49, 65] |
| Unidirectional formula (Eq. S3) | 25 days | 59 days [51, 66] |
| Share of the peak captured by one quarter | 32% | **89%** [81, 94] |

S14's 25 days, "a third of the signal" and the oscillatory regime all depend on the biased autoregressions. Under latent centering, the reciprocal formula applies, the optimum is about two months, and quarterly sampling captures about 90% of the peak.

## Convergence

- Real latent + stationary: R-hat ≤ 1.004, bulk ESS ≥ 1,062, BFMI ≥ 0.57, 18 divergences out of 16,000 draws.
- Sim 2 stationary had 539 divergences, though R-hat (1.003) and ESS (≥ 1,350) were fine. Treat that replicate with some caution.
- The stationary fits start every chain at zero transition terms, with no jitter (`init="adapt_diag"`). The default random start makes some persons' dynamics explosive, so every chain failed to initialize. This override is recorded in each fit's diagnostics.
- Real latent with the conditioned first point (not recommended): 1,484 divergences, bulk ESS 166, BFMI 0.11. It is in `latent_vs_published.csv` as the `latent_` columns, for completeness only.

## Choices to note

- The localization contrast K stays centered on its observed mean (it is only a predictor). The interactions are rebuilt from the latent deviations.
- Usual levels are independent of the random slopes. The correlation prior is Beta(2, 2), which is identical to LKJ(eta = 2) in two dimensions.
- The stationary distribution uses each person's transition matrix at K = 0.
- Standardized couplings use step 08's within-person SDs, unchanged.
- All recovery fits ran at the full sampler settings (4,000/4,000, 4 chains, target_accept 0.99), so the sampler was not reduced anywhere.
