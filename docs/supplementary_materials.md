Supplementary Materials: Bidirectional Quarterly Sleep-Pain Coupling in Knee Osteoarthritis

## Factor analysis and pain localization contrast validation

![](../results/step2_contrast_validation/step2_figure_s1_endorsement.png)

**Figure S1.** Pain localization contrast factor and PHQ body map endorsements. **(A)** Point-biserial correlations between endorsement of each of the 13 PHQ body map pain areas and the person-mean pain localization contrast factor ($\bar{K}_{i}$). Blue = knee; coral = non-knee areas. Black border indicates FDR-corrected significance ($q<0.05$). Only knee endorsement survived correction; all 12 non-knee areas showed negative or near-zero associations. **(B)** Person-mean contrast ($\bar{K}_{i}$) by pain distribution group defined from PHQ body map endorsements. Knee only: endorsed knee pain but no other areas (N = 19); Knee + others: endorsed knee pain plus at least one other area (N = 143); No knee: did not endorse knee pain (N = 67). Horizontal lines indicate significant Tukey post-hoc comparisons. One-way ANOVA: $F(2,226)=16.56$, $p<0.001$.

![](../results/step2_contrast_validation/step2_figure_s2_convergent.png)

**Figure S2.** Convergent validity of the pain localization contrast factor. Scatter plots showing the relationship between each participant's mean contrast score across all available quarters ($\bar{K}_{i}$) and baseline clinical measures not used in the factor analysis. Pearson correlations are shown for continuous measures; Spearman $\rho$ for the ordinal Kellgren-Lawrence grade. All knee-specific measures correlate positively with the contrast factor, confirming that higher contrast scores reflect greater predominance of knee-localized relative to body-wide pain.

## Pain localization moderation of coupling

![](../results/step5_contrast_moderation/step5_figure_s3_jn_localization_sp.png)

**Figure S3.** Johnson-Neyman analysis of pain localization moderation of sleep-to-pain coupling. The blue line shows the posterior mean coupling slope as a continuous function of within-person pain localization ($K^{w}$), dashed lines show the 95% credible interval, and grey shading indicates that the CrI includes zero across the entire observed range (no JN boundary). Vertical markers show simple slopes at body-dominant (-2 SD), balanced (0), and knee-dominant (+2 SD) localization levels with 95% CrI error bars. Blue dots show fitted coupling values (observation-level).

## Sleep-to-pain fMRI moderation

![](../figures/figure_s4_stim_rois_2col.png)

**Figure S4.** Spherical regions of interest for sleep-to-pain coupling moderation, shown on orthogonal slices of the MNI152 T1 template. Each panel displays one ROI with sagittal, coronal, and axial views centered on the sphere. Top five ROIs from the Krause et al. (2019) sleep deprivation framework: Contralateral Somatosensory Cortex (S1; MNI ±36, -45, 59; r = 8 mm; hemisphere contralateral to stimulated knee), Contralateral Middle Insula (±32, 4, 11; 8 mm; contralateral to stimulated knee), Left Thalamus (-10, -6, 10; 4 mm), Left Anterior Insula (-27, 25, 0; 8 mm), and Left and Right Nucleus Accumbens (±9, 2, -7; 6 mm; tested separately). Bottom ROI from the Sardi et al. (2024) framework: Right dACC/MCC (6, 12, 38; 6 mm), defined from the Xu et al. (2020) pain fMRI meta-analysis.

![](../results/step9_sp_jn/step9_figure_s5_krause_jn.png)

**Figure S5.** Johnson-Neyman analyses of non-significant Krause ROI moderation of sleep-to-pain coupling ($\gamma_{sp}$). **(A)** Contralateral Somatosensory Cortex (S1). **(B)** Contralateral Middle Insula. **(C)** Left Thalamus. **(D)** Left Anterior Insula. For each panel, the blue line shows the posterior mean coupling slope as a continuous function of ROI activation (z-scored), dashed lines show the 95% credible interval. Vertical markers show simple slopes at low (Q1 - $1.5 \times \mathrm{IQR}$), median, and high (Q3 + $1.5 \times \mathrm{IQR}$) levels with 95% CrI error bars. Blue dots show person-level fitted coupling values (population-level slope + random effect). N = 173 for contralateral S1 and contralateral middle insula (one participant missing stimulation site); N = 174 for thalamus and anterior insula.

## Pain-to-sleep arousal relay moderation

![](../figures/figure_s8_arousal_rois_2col.png)

**Figure S6.** Atlas-defined probabilistic regions of interest corresponding to nodes of the Lynch et al. (2025) pain-arousal relay pathway, shown on orthogonal slices of the MNI152 T1 template. Each panel displays one bilateral ROI with sagittal, coronal, and axial views centered on the atlas center of mass. Lateral Parabrachial Nucleus (PBN; Brainstem Navigator atlas (Singh et al., 2022); 3 brain voxels at 3 mm fMRI resolution), Substantia Innominata / Basal Forebrain (SI-BF/Ch4; Zaborszky et al. (2008) cytoarchitectonic atlas), Central Nucleus of the Amygdala (CeA; CIT168 atlas (Pauli et al., 2018)), Bed Nucleus of the Stria Terminalis (BNST; Theiss et al. (2017) atlas), and Lateral Hypothalamus (LH; Neudorfer et al. (2020) atlas). Atlas masks are shown as resampled to 1 mm resolution; probability-weighted extraction was performed at fMRI (3 mm) and VBM (1.5 mm) resolution (see Methods).

**Table S1.** Pain-arousal relay moderation of pain-to-sleep coupling (atlas-defined ROIs).

| ROI       | Modality      | $\gamma_{ps}$ | 95% CrI          |     $p$ |
| :-------- | :------------ | ------------: | :--------------- | ------: |
| PBN       | fMRI response |        -0.075 | [-0.175, +0.025] |   0.145 |
|           | GM volume     |        -0.062 | [-0.173, +0.046] |   0.261 |
| BNST      | fMRI response |        -0.077 | [-0.191, +0.031] |   0.166 |
|           | GM volume     |        -0.039 | [-0.145, +0.067] |   0.474 |
| CeA       | fMRI response |        +0.073 | [-0.044, +0.189] |   0.212 |
|           | GM volume     |        -0.050 | [-0.167, +0.064] |   0.398 |
| SI-BF/Ch4 | fMRI response |        +0.022 | [-0.078, +0.117] |   0.660 |
|           | GM volume     |        -0.018 | [-0.132, +0.095] |   0.761 |
| LH        | fMRI response |        -0.022 | [-0.124, +0.078] |   0.666 |
|           | GM volume     |        -0.014 | [-0.120, +0.090] |   0.798 |

**Note.** Each ROI was tested in a separate model run. fMRI response: N = 174; unmasked contrast images were used for PBN, SI-BF/Ch4, CeA, and BNST; GM-masked contrasts for LH (see Methods). GM volume: N = 189. $p = 2 \times \min(P(\gamma_{ps} < 0),\, P(\gamma_{ps} > 0))$, computed from the posterior distribution. Johnson-Neyman analyses for each ROI are shown in Figures S7 (fMRI BOLD) and S8 (GM volume). Sleep-to-pain moderation results for all six spherical ROIs (including ACC) are reported in Table 5 of the main text.

![](../results/step12_ps_jn/step12_figure_s7_fmri_arousal_jn.png)

**Figure S7.** Johnson-Neyman analyses of fMRI BOLD moderation of pain-to-sleep coupling ($\gamma_{ps}$) for five pain-arousal relay ROIs. **(A)** Parabrachial Nucleus (PBN). **(B)** Substantia Innominata / Basal Forebrain (SI-BF/Ch4). **(C)** Central Nucleus of the Amygdala (CeA). **(D)** Bed Nucleus of the Stria Terminalis (BNST). **(E)** Lateral Hypothalamus (LH). For each panel, the blue line shows the posterior mean coupling slope as a continuous function of ROI activation (z-scored), dashed lines show the 95% credible interval. Vertical markers show simple slopes at low (Q1 - $1.5 \times \mathrm{IQR}$), median, and high (Q3 + $1.5 \times \mathrm{IQR}$) levels with 95% CrI error bars. Blue dots show person-level fitted coupling values (population-level slope + random effect). N = 174.

![](../results/step12_ps_jn/step12_figure_s8_vbm_arousal_jn.png)

**Figure S8.** Johnson-Neyman analyses of grey matter volume moderation of pain-to-sleep coupling ($\gamma_{ps}$) for five pain-arousal relay ROIs. **(A)** Parabrachial Nucleus (PBN). **(B)** Substantia Innominata / Basal Forebrain (SI-BF/Ch4). **(C)** Central Nucleus of the Amygdala (CeA). **(D)** Bed Nucleus of the Stria Terminalis (BNST). **(E)** Lateral Hypothalamus (LH). Format as in Figure S7 but using probability-weighted GM integral (mm³) from published atlases at 1.5 mm VBM resolution. N = 189.

## Severity moderation of coupling

**Table S2.** Person-mean severity moderation of coupling.

| Moderator          | Model | Direction | $\gamma$ | 95% CrI          |     $p$ |
| :----------------- | :---- | :-------: | -------: | :--------------- | ------: |
| Mean Pain Severity | Alone |    SP     |   -0.010 | [-0.047, +0.028] |   0.611 |
| Mean Sleep Quality | Alone |    SP     |   +0.023 | [-0.017, +0.064] |   0.267 |
| Mean Pain Severity | Joint |    SP     |   -0.003 | [-0.042, +0.035] |   0.877 |
| Mean Sleep Quality | Joint |    SP     |   +0.023 | [-0.020, +0.065] |   0.301 |

**Note.** Each moderator was z-scored. Three models were run: person-mean pain severity alone, person-mean sleep quality alone, and both simultaneously (joint). N = 229; 1,818 observations. $p = 2 \times \min(P(\gamma < 0),\, P(\gamma > 0))$. None of the moderation parameters approached significance (all $p > 0.25$, all 95% CrIs comfortably including zero), indicating that within-person coupling operates comparably regardless of baseline severity.

## Supplementary Note S1: Additional Methodological Considerations

The small proportion of individually credible coupling estimates (9/229 for pain-to-sleep, 4/229 for sleep-to-pain) reflects the partial pooling inherent in hierarchical Bayesian estimation rather than an absence of true individual differences. With a median of 9 quarterly observations per participant, individual-level data are necessarily noisy, and the hierarchical model concentrates inferential power at the population level—where the pooled data across 229 participants and 1,818 transitions enable reliable estimation of mean coupling, heterogeneity, and moderating effects. The individual random effects $u_{i}$ serve primarily to partition variance and improve population-level estimates rather than to provide precise person-level inference.

Relatedly, although the pain-to-sleep coupling ($|\hat{\lambda}_{ps}| = 0.140$) qualifies as a large cross-lagged effect by the benchmarks of Orth et al. (2024) (small = 0.03, medium = 0.07, large = 0.12), its absolute magnitude is modest in the context of noisy quarterly within-person deviations, and the large between-person heterogeneity ($\hat{\tau}_{ps} = 0.362$) indicates that the population mean reflects a meaningful but noisy signal, consistent with the expectation that quarterly assessments introduce variability from sources unrelated to the sleep-pain dynamic.

The autoregressive coefficients were notably small ($\hat{\phi}_{p} = 0.106$, $\hat{\phi}_{s} = 0.004$). Among the 16 daily studies reviewed, only Edwards et al. (2008) includes explicit autoregressive terms as fixed-effect predictors (pain AR = 0.18, sleep AR = 0.15); a few others handle serial dependence through residual covariance structures (6, 7, 11), and the majority omit autoregressive control entirely. While retaining them is more principled—omitting the lagged dependent variable can inflate cross-lagged estimates when the outcome is autocorrelated (63)—the negligible magnitudes observed here suggest that simpler models without autoregressive terms may yield equivalent coupling estimates in quarterly designs, where the long interval between assessments attenuates day-to-day persistence.

Relatedly, the quarterly temporal resolution may be suboptimal for the coupling process we are detecting. The Dormann and Griffin (2015) framework provides a principled approach to optimal measurement interval: the eigenvalues of the bivariate transition matrix yield timescale-invariant continuous-time drift parameters (**Supplementary Note S2**). The dominant eigenvalue ($\lambda_{1} \approx 0.130$) exceeds the larger autoregression ($\hat{\phi}_{p} = 0.106$), confirming that reciprocal coupling stabilizes the system. However, because the product of the cross-lagged coefficients exceeds the product of the autoregressions ($\hat{\lambda}_{sp}\hat{\lambda}_{ps} = 0.00294 > \hat{\phi}_{s}\hat{\phi}_{p} = 0.000424$), the minor eigenvalue is negative ($\lambda_{2} \approx -0.020$), placing the system in an oscillatory regime where the closed-form optimal lag is undefined (63). A unidirectional reference calculation (setting $\lambda_{ps} = 0$) yields an optimal lag of approximately 0.275 quarters ($\approx 25$ days), with quarterly measurement capturing roughly 32% of the peak unidirectional cross-lagged signal (**Supplementary Note S2**). Weekly or biweekly measurements would substantially increase sensitivity to these coupling processes while restoring the monotonic regime in which the reciprocal optimal-lag formula applies. The coupling patterns observed here reflect cumulative quarterly processes and should not be directly compared with daily diary findings.

## Supplementary Note S2: Optimal Time Lags in the Bivariate Sleep-Pain System

### Overview

Dormann and Griffin (2015) showed that the observable cross-lagged effect in a panel design follows a non-monotonic curve as a function of the measurement interval: it rises from zero, peaks at an *optimal lag*, then declines. This note derives the relevant formulas for the reciprocal (bidirectional) case, recovers timescale-invariant continuous-time drift parameters, and applies them to the present estimates. The framework assumes that a perturbation in one variable affects the other only insofar as it persists through autoregressive dynamics. If a transient perturbation instead triggers a biological cascade with its own timescale, the formulas below do not apply to that component.

### Setup

The bivariate VAR(1) for within-person deviations, ignoring exogenous terms and innovations, is

$$\mathbf{z}_{t} = \mathbf{B}\,\mathbf{z}_{t-1},$$

with state vector and transition matrix

$$\mathbf{z}_{t} = \begin{pmatrix} S_{t}^{w} \\ P_{t}^{w} \end{pmatrix}, \qquad \mathbf{B} = \begin{pmatrix} \phi_{s} & \lambda_{ps} \\ \lambda_{sp} & \phi_{p} \end{pmatrix}.$$

All coefficients are at a specified unit (SU) of one quarter ($\approx 91$ days). The total cross-lagged effect of $S_{0}^{w}$ on $P_{\omega}^{w}$ (controlling for $P_{0}^{w}$) at lag $\omega$ (in SU multiples) is the $(2,1)$ element of $\mathbf{B}^{\omega}$.

### Eigenvalues and the cross-lagged effect

The characteristic equation $(\phi_{s} - \lambda)(\phi_{p} - \lambda) - \lambda_{sp}\lambda_{ps} = 0$ yields

$$\lambda_{1} = \frac{\phi_{s} + \phi_{p} + \Delta}{2}, \qquad \lambda_{2} = \frac{\phi_{s} + \phi_{p} - \Delta}{2},$$

with discriminant

$$\Delta = \sqrt{(\phi_{p} - \phi_{s})^{2} + 4\,\lambda_{sp}\,\lambda_{ps}}.$$

These eigenvalues are the *effective stabilities* of the coupled system. When both coupling coefficients share the same sign, $\lambda_{sp}\lambda_{ps} > 0$, which increases $\Delta$ and pushes $\lambda_{1}$ above $\max(\phi_{s}, \phi_{p})$: reciprocal coupling stabilizes the dominant mode.

Via eigendecomposition ($\mathbf{B}^{\omega} = \mathbf{P}\,\mathrm{diag}(\lambda_{1}^{\omega}, \lambda_{2}^{\omega})\,\mathbf{P}^{-1}$) and exploiting the identity $(\lambda_{1} - \phi_{s})(\lambda_{2} - \phi_{s}) = -\lambda_{sp}\lambda_{ps}$, the $(2,1)$ element simplifies to

$$\beta(\omega) = \lambda_{sp} \cdot \frac{\lambda_{1}^{\omega} - \lambda_{2}^{\omega}}{\lambda_{1} - \lambda_{2}}. \tag{S1}$$

The reverse direction has the same temporal profile: $[\mathbf{B}^{\omega}]_{1,2} = \lambda_{ps} \cdot (\lambda_{1}^{\omega} - \lambda_{2}^{\omega})/(\lambda_{1} - \lambda_{2})$. Both peak at the same lag.

### Optimal lag

Setting $\partial \beta / \partial \omega = 0$ gives $\lambda_{1}^{\omega} \ln \lambda_{1} = \lambda_{2}^{\omega} \ln \lambda_{2}$, which solves to

$$\omega_{\mathrm{opt}} = -\frac{\ln\!\left(\dfrac{\ln \lambda_{1}}{\ln \lambda_{2}}\right)}{\ln \lambda_{1} - \ln \lambda_{2}}. \tag{S2}$$

This requires both eigenvalues to be positive. When $\lambda_{2} < 0$, $\ln \lambda_{2}$ is complex and no real-valued optimal lag exists.

When the reciprocal formula is inapplicable, a unidirectional reference calculation (setting $\lambda_{ps} = 0$, so that $\mathbf{B}$ becomes triangular with eigenvalues $\lambda_{1} \to \phi_{p}$ and $\lambda_{2} \to \phi_{s}$) provides a tractable surrogate:

$$\omega_{\mathrm{opt}}^{\,\mathrm{unidir}} = -\frac{\ln\!\left(\dfrac{\ln \phi_{p}}{\ln \phi_{s}}\right)}{\ln \phi_{p} - \ln \phi_{s}}. \tag{S3}$$

The corresponding signal attenuation at any lag $\omega$ relative to the optimal lag is given by the ratio of the amplification factors:

$$\frac{\beta(\omega)}{\beta(\omega_{\mathrm{opt}})} = \frac{f(\omega)}{f(\omega_{\mathrm{opt}})}, \qquad f(\omega) = \frac{\phi_{p}^{\,\omega} - \phi_{s}^{\,\omega}}{\phi_{p} - \phi_{s}}. \tag{S4}$$

### Continuous-time drift

The discrete transition matrix $\mathbf{B}$ relates to a continuous-time drift matrix $\mathbf{A}$ via $\mathbf{B} = e^{\mathbf{A}\,\Delta t}$, where $\Delta t$ is the SU in real time. The eigenvalues of $\mathbf{A}$ are

$$\alpha_{k} = \frac{\ln \lambda_{k}}{\Delta t}, \qquad k = 1, 2. \tag{S5}$$

These are timescale-invariant: the effective stability at any alternative lag $\Delta t'$ is $\lambda_{k}(\Delta t') = \exp(\alpha_{k} \Delta t')$. When $\lambda_{2} < 0$, $\alpha_{2}$ is complex, reflecting the oscillatory mode.

### Application

**1. Eigenvalues.** From Table 3: $\hat{\phi}_{p} = 0.106$, $\hat{\phi}_{s} = 0.004$, $\hat{\lambda}_{sp} = -0.021$, $\hat{\lambda}_{ps} = -0.140$. Then

$$\Delta = \sqrt{(0.106 - 0.004)^{2} + 4(-0.021)(-0.140)} = \sqrt{0.02216} \approx 0.149,$$

$$\lambda_{1} \approx 0.130, \qquad \lambda_{2} \approx -0.020.$$

The dominant eigenvalue exceeds $\hat{\phi}_{p}$ by 22%, confirming stabilization through reciprocal coupling ($\hat{\lambda}_{sp}\hat{\lambda}_{ps} = 0.00294 > 0$).

**2. Continuous-time drift.** For the dominant mode,

$$\alpha_{1} = \frac{\ln(0.130)}{1\;\mathrm{quarter}} \approx -2.04\;\mathrm{quarter}^{-1} \approx -0.022\;\mathrm{day}^{-1}.$$

The dominant-mode stability at alternative lags is weekly $e^{-2.04/13} \approx 0.85$ and monthly $e^{-2.04/3} \approx 0.51$.

**3. Oscillatory regime.** The minor eigenvalue is negative ($\lambda_{2} \approx -0.020$) because the product of the cross-lagged coefficients exceeds the product of the autoregressions: $\hat{\lambda}_{sp}\hat{\lambda}_{ps} = 0.00294 > \hat{\phi}_{s}\hat{\phi}_{p} = 0.000424$. Equivalently, $\Delta > \hat{\phi}_{s} + \hat{\phi}_{p}$, which drives $\lambda_{2}$ below zero. This produces oscillatory dynamics in $\beta(\omega)$—the term $\lambda_{2}^{\omega}$ alternates sign at successive integer lags—and renders the reciprocal optimal lag (Eq. S2) undefined because $\ln \lambda_{2}$ is complex. Shorter measurement intervals (weekly or biweekly) would likely restore $\lambda_{sp}\lambda_{ps} < \phi_{s}\phi_{p}$, making the reciprocal formula applicable.

**4. Unidirectional approximation and signal attenuation.** Applying Eq. S3,

$$\omega_{\mathrm{opt}}^{\,\mathrm{unidir}} = -\frac{\ln\!\left(\dfrac{\ln(0.106)}{\ln(0.004)}\right)}{\ln(0.106) - \ln(0.004)} = -\frac{\ln(0.406)}{3.277} \approx 0.275\;\mathrm{quarters} \approx 25\;\mathrm{days}.$$

Applying Eq. S4 at $\omega = 1$ (one quarter) versus $\omega_{\mathrm{opt}} = 0.275$,

$$f(1) = \frac{0.106 - 0.004}{0.102} = 1.00, \qquad f(0.275) = \frac{0.106^{0.275} - 0.004^{0.275}}{0.102} \approx 3.14.$$

The quarterly measurement therefore captures approximately $1.00 / 3.14 \approx 32\%$ of the peak unidirectional cross-lagged signal. Weekly or biweekly intervals would recover a substantially larger fraction.