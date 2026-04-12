# Manuscript Corrections Log

Tracking every correction, tension, or known issue that needs to be
resolved in the manuscript docx. Items are added as they are
discovered in the course of building the new `new_organization/`
pipeline. Each entry states what the manuscript currently says,
what the code actually does (or should do), and the recommended
edit.

---

## 1. Methods §2.3 — Parallel-analysis threshold for Component 1

**Status:** pending — final value depends on the reimplementation
of the factor analysis in the new pipeline (Step 1). The numbers
below describe what was true in the session that produced the
published draft; they may change once Step 1 runs on the new
Step 0 output.

**Manuscript text (line 231):**
> "The first factor was dominant (eigenvalue = 5.56, 69.5% of
> variance) and exceeded the 95th-percentile random eigenvalue
> threshold (**1.13**)."

**Problem.** The parallel analysis was run in an earlier session
(2026-03-27) and produced both a 95th-percentile row and a random
mean row:

  Component 1: actual = 5.557, Random 95th = 1.171, Random Mean = 1.131
  Component 2: actual = 0.890, Random 95th = 1.115, Random Mean = 1.084

For Component 1 the manuscript cites **1.13**, which is the random
**mean** (1.131), while calling it the "95th-percentile". For
Component 2 the manuscript cites **1.11**, which is correctly the
95th percentile.

**Candidate correction.** Change Comp 1's threshold from `1.13` to
`1.17` so both components use the same (95th-percentile) reference.
The substantive conclusion is unaffected: 5.56 vastly exceeds 1.13
and 1.17 alike. But defer the final edit until Step 1 rewrites the
factor analysis from scratch on the new Step 0 output, so that
whatever values go into the manuscript come directly from a
committed, runnable script.

---

## 2. Table 1 — Reduce to quarterly items used in the factor analysis

**Manuscript current state:** Table 1 lists all 25 quarterly items
from the questionnaire (q1-q25), even though only 8 of them
(q2-q5 for knee, q7-q10 for body) feed the factor analysis, a 9th
(q13) is the sleep quality item, and q1 and q6 are the gateway
questions used for imputation. The remaining items — q11 (fatigue),
q12 (mood), q14-q15 (treatment), q16-q25 (PSS perceived-stress
items) — are not used in any modeling and clutter the table.

**Required edit.**
1. In the main text Table 1, keep only the items that are actually
   modeled: q1, q2, q3, q4, q5 (knee gateway + intensity +
   interference), q6, q7, q8, q9, q10 (body gateway + intensity +
   interference), and q13 (sleep quality).
2. Move the full 25-item list to the supplementary materials as a
   new table (e.g., "Table S1 — Complete quarterly questionnaire
   items"). Preserve the exact wording from the data dictionary.
3. Add a sentence in Methods §2.2 (or wherever Table 1 is first
   introduced) noting that the questionnaire contains additional
   items (fatigue, mood, treatment, perceived stress) that are
   not used in the present analysis, with a pointer to
   Supplementary Table S1.

---

## 3. Gateway imputation scope — updated to 4+4

**What the earlier draft of the paper did.** The committed
`scripts/prepare_data_contrast.py` gated only the three intensity
items in each region (q2/q3/q4 for knee, q7/q8/q9 for body), then
ran the factor analysis on all eight items (q2-q5, q7-q10). The
interference items q5 and q10 were left as NaN when the gateway
question said "no pain", which made those rows partial-item
Bartlett scores.

**Why this is wrong.** If a subject says "no knee pain this week"
at q1, then the rating of how much knee pain interfered with
general activity (q5) is also structurally 0, not missing. Same
for body pain (q6 -> q10). Using ungated items in a factor
analysis is logically inconsistent with the gateway structure of
the questionnaire.

**What the new pipeline does.** Step 0 of `new_organization/`
applies 4+4 gating:

  knee: q1 == 0 -> q2, q3, q4, q5 -> 0 when NaN
  body: q6 == 0 -> q7, q8, q9, q10 -> 0 when NaN

The same eight items (q2-q5, q7-q10) continue to feed the factor
analysis in Step 1; all that changes is that 48 extra item values
on 34 person-quarters flip from NaN to 0.

**Verified consequences (Steps 0–2 re-run 2026-04-11).**
The full pipeline with 4+4 gating reproduces the sample-size chain
exactly:

  Parent study N:       243 (matches paper)
  Excluded (no ≥3 seg): 14  (matches paper)
  Analytic sample:      229 (matches paper)
  Retained points:      2,056 (matches paper)
  Lag transitions:      1,818 (matches paper)
  Median lags/person:   9, range [2, 10] (matches paper)

The only change is the number of interpolated retained points:
**113** (was 128). This is because 4+4 gating fills more raw items
upfront, so fewer factor-score gaps remain to be interpolated
downstream.

**Required manuscript edit.** Update the Figure 1 caption where it
says "128 of 2,056 retained points were computed from partial raw
items" to **113 of 2,056**. All other counts (N, retained,
transitions, median, range) are unchanged.

---

## 4. Results §3.1 — Factor analysis numbers need updating

Step 1 of the new pipeline (4+4 gating) produces different factor
analysis numbers from the manuscript. All of these need to be
updated in Results §3.1 (the "Factor analysis and pain localization
contrast" subsection).

| Metric | Manuscript | New pipeline |
| --- | --- | --- |
| F1 eigenvalue | 5.56 | **5.51** |
| F2 eigenvalue | 0.89 | **1.45** |
| F1 % variance | 69.5% | **68.9%** |
| F2 % variance | 11.2% | **18.1%** |
| Total 2-factor % variance | 80.7% | **87.0%** |
| PA 95th Comp 1 | 1.13 (wrong — was random mean) | **1.12** |
| PA 95th Comp 2 | 1.11 | **1.08** |
| F2 passes PA? | no | **YES** |
| F1 loadings range | 0.78–0.86 | 0.78–0.87 |
| F2 knee loadings | +0.30 to +0.45 | +0.30 to +0.46 |
| F2 body loadings | -0.23 to -0.50 | -0.26 to -0.52 |
| r(F1, F2) | 0.001 | -0.007 |
| Interpolated points | 128 | **118** |

**Key narrative change.** The manuscript currently says: "The
second eigenvalue (0.89, 11.2%) fell below the parallel analysis
threshold (1.11), indicating that a one-factor solution would be
sufficient on purely statistical grounds. Nevertheless, a two-factor
solution was retained because..." Under the new pipeline, F2 (1.45)
exceeds the PA 95th-percentile threshold (1.08), so **both factors
pass parallel analysis**. The "nevertheless" sentence should be
replaced with a straightforward statement that both factors are
retained on statistical and theoretical grounds.

**Also add:** The manuscript currently does not report the total
variance explained by the 2-factor solution. Add it: "The retained
two-factor solution jointly accounted for **87.0%** of the
variance" (replacing the current 80.7%).

---

## 6. Table 3 — Race/ethnicity categories

The data dictionary labels `qst_bedside_race_ethno___1__s1` as
"Asian or Asian American", but the endorsement count (N=55) matches
the manuscript's "Hispanic/Latino" category. Likely a REDCap
coding/labeling mismatch. Not important for the analysis — race is
only a descriptive variable in Table 3, not used in modeling.
Verify with the study team if time permits; otherwise keep the
current Table 3 output as-is (91 Black, 68 White, 55 labeled per
dictionary, remainder as Other).

---

## 7. Step 3 — Coupling parameters and LOO-CV

All Table 4 population parameters reproduce within |Δ| ≤ 0.005
(MCMC noise) of the manuscript values. All inferential conclusions
are identical. The exact point estimates need to be updated to
the new pipeline values in the manuscript.

LOO-CV Δ/SE values are essentially unchanged:
  full vs no_PS: 2.48 (was 2.49)
  no_SP vs null: 2.41 (was 2.53)
  full vs no_SP: 0.04 (was 0.12)
  no_PS vs null: 0.02 (was 0.30)

Both PS tests still cross |Δ/SE| > 2; both SP tests still do not.

Person-level statistics match exactly: PS SD = 0.180, N credible = 9;
SP SD = 0.049, N credible = 4.

**Pareto k-hat improved:** max = 0.66 (was 0.90), 0/1818 above 0.7
(was 2/1818). The manuscript sentence "only 2 of 1,818 observations
(0.1%) exceeded the 0.7 threshold (maximum k-hat = 0.90)" should
be updated to "no observations exceeded the 0.7 threshold (maximum
k-hat = 0.66)".

---

## 8. Table 5 is redundant — merge into Table 4

The manuscript's Table 5 ("Contrast moderation parameters") lists
delta_p, omega_sp, delta_s, omega_ps — all four of which are
already rows in Table 4. Remove Table 5 from the manuscript and
renumber Table 6 → Table 5. This also means all references to
"Table 5" and "Table 6" in the text need to be shifted by one.

---

## 9. Step 4 — Contrast moderation JN

JN boundary for PS coupling shifted from -0.625 (-0.86 SD) to
**-0.602 (-0.83 SD)**. Percentage in credible region: 84.4%
(was 84.6%). All simple-slope conclusions unchanged. SP direction
still null (no boundary). Update the boundary value and SD in the
Results text and Figure 4 caption.

---

## 10. Steps 5–7 — SP moderation (Table 5, Figures 5/6/S5)

All γ_sp estimates reproduce within |Δ| ≤ 0.003 of the manuscript.
Both headline findings remain credible:
  Left NAcc: p = 0.032 (was 0.027)
  Right dACC/MCC: p = 0.042 (was 0.047)
Sign concordance: 6/6, p = 0.0156 (unchanged).
NAcc-ACC correlation: r = 0.12 (exact match).

JN boundaries shifted (MCMC noise):
  Left NAcc raw boundary: 0.001 (was -0.027), 52.9% below (was 49%)
  ACC raw boundary: 0.119 (was 0.062), 55.2% below

Update the exact γ_sp point estimates, p-values, and JN boundaries
in Table 5 (now Table 5 after the old Table 5 was merged into
Table 4), Figures 5 and 6 captions, and the Results text.

Note: N = 174 for all ROIs — **unchanged** from the paper, because
the fMRI subsample depends on who has neuroimaging data, not on
the factor-score gating change. The paper says N = 174 and we
reproduce it exactly.

---

## 11. Steps 8–10 — PS arousal moderation (Table S1, Figures S7/S8)

fMRI BOLD (N=174 ✓) and VBM GM volume (N=189 ✓) sample sizes match
the paper exactly.

VBM: all 5 gamma_ps within |Δ| ≤ 0.004. Sign concordance 5/5,
p = 0.0312 (unchanged). All null.

fMRI: SI-BF/Ch4, CeA, BNST within |Δ| ≤ 0.007. LH shifted from
-0.022 to -0.002 (both null). PBN sign-flipped (+0.026 vs -0.075),
known 2-vs-3 voxel issue at fMRI resolution. All null.

Update the exact gamma_ps values and p-values in Table S1 and the
Results text. No inferential conclusions change.

Note: VBM filenames use `x` instead of `-` in subject IDs
(BIDS convention); Step 8 maps `x` → `-` to match the quarterly
data IDs.

---

## 5. Results — Clarify the order of data-preparation operations

The manuscript's Results section presents Figure 1 (data
availability grid showing observed, interpolated, and retained
person-quarters) alongside the factor-analysis results, but does
not make the order of operations clear to the reader. The actual
pipeline order is:

  1. Gateway imputation (raw items, Step 0)
  2. Factor analysis + Bartlett scoring (Step 1)
  3. Interpolation of single interior gaps in factor scores (Step 1)
  4. Segment filter (≥3 consecutive quarters, Step 2)
  5. Within-person decomposition + lag creation (Step 2)
  6. Figure 1 generated from the segment-filter results (Step 2)

The reader could misunderstand Figure 1 as showing the raw
quarterly data availability before any processing, when in fact
the "interpolated" dots and the "retained" segments in the figure
depend on Steps 1–2 having already run. The Methods or Results
text should include a brief statement making the processing
sequence explicit, so the reader knows what Figure 1 is actually
showing.

---

