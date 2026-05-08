# GCPS as a convergent-validity reference for the knee–body pain contrast

## Background

The Graded Chronic Pain Scale (GCPS; Von Korff, Ormel, Keefe, & Dworkin, 1992) is a brief seven-item self-report instrument widely used in chronic pain research. It produces two summary scores, each rescaled from the original 0–10 ratings to a common 0–100 metric:

- **Characteristic pain intensity** — the mean of three 0–10 items rating *current pain*, *worst pain in the last six months*, and *average pain in the last six months*, multiplied by 10. It captures the typical severity of pain experienced by the participant over the recall window, independent of how disabling that pain is.
- **Pain-related interference (disability score)** — the mean of three 0–10 items rating how much pain has interfered with *daily activities*, *recreational/social/family activities*, and *the ability to work or do housework* over the last six months, multiplied by 10. It captures the functional impact of pain rather than its raw intensity.

The two scores are conceptually distinct (intensity vs. impact) but are typically moderately correlated, and together they support the GCPS chronic-pain-grade classification. In the present study, GCPS scores were collected at the baseline session through the standard REDCap form, and the pre-computed `gcps_pain_intensity` and `gcps_interference` variables were extracted from the wide-format dataset.

## Hypothesis

If the knee–body pain contrast factor (F2) recovered from the quarterly factor analysis really does index the *localization* of pain to the knee — i.e. the degree to which a participant's reported pain is concentrated at the knee rather than spread across the body — then participants with a higher person-mean contrast score (K̄) should also report more clinically meaningful knee-related chronic pain on a standard, externally validated chronic-pain instrument. We therefore tested whether K̄ correlates positively with both GCPS subscales. This complements the existing convergent-validity panel for K̄ against WOMAC, PHQ, KL grade, and the QST knee-pain rating.

## Methods

K̄ was defined as in step 02 of the analysis pipeline: each participant's mean contrast-factor score across all available quarters. GCPS scores were taken from the baseline session and matched to participants by ID. Pearson product–moment correlations were computed between K̄ and each GCPS subscale. The analytic sample was N = 241 — every participant in the quarterly cohort with a non-missing baseline GCPS.

## Results

| Reference measure | r | p | N |
|---|---:|---:|---:|
| GCPS characteristic pain intensity (0–100) | +0.262 | < 0.001 | 241 |
| GCPS pain-related interference (0–100) | +0.259 | < 0.001 | 241 |

Both correlations were significantly positive at the conventional 0.05 level (and at 0.001), in the direction predicted by the localization interpretation: participants whose quarterly pain reports were more knee-concentrated (higher K̄) reported both more intense baseline chronic pain and more pain-related functional interference.

The effect sizes (r ≈ +0.26 for both subscales) are in the same range as the convergent-validity correlations already reported for K̄ against the WOMAC subscales, the PHQ knee-pain items, and the QST knee-pain rating, confirming that the contrast factor behaves like a knee-pain severity proxy without being redundant with any single clinical scale. The near-identical magnitudes for intensity and interference are consistent with the expected moderate-to-high correlation between the two GCPS subscales themselves and indicate that K̄ tracks the shared pain-burden component captured by both subscales rather than disability or intensity in isolation.

## Conclusion

These two new correlations strengthen the external validation of the knee–body contrast factor: K̄ is positively and significantly associated with both the characteristic pain intensity and the pain-related interference subscales of the GCPS, a standard chronic-pain instrument. This supports the interpretation of K̄ as a localization index that carries clinically meaningful chronic-pain information.
