# Quarterly Sleep-Pain Coupling in Knee Pain: Pain-to-Sleep Dominance and NAcc-Gated Sleep-to-Pain

*\*Corresponding author:*

Pedro A. Valdes-Hernandez

Department of Health Outcomes and Biomedical Informatics

College of Medicine, University of Florida

1889 Museum Road, Suite 7018, Gainesville, FL 32611, USA.

Email: <pvaldeshernandez@ufl.edu>

Phone: +1 (352) 294-5984

# Abstract

Sleep and pain are bidirectionally related, yet despite consistent day-to-day evidence, within-person dynamics at longer timescales remain poorly understood. We examined bidirectional sleep-pain coupling quarterly in adults from a knee pain study (N=229; up to 11 time points). Pain and sleep numerical rating scales were collected; eight pain items were factor-analyzed into general severity and a knee-versus-body contrast. A Bayesian hierarchical bivariate model with person-specific coupling slopes was fit. Population-level pain-to-sleep coupling was credibly negative, indicating the dominant direction, with substantial between-person heterogeneity and the only direction whose inclusion improved out-of-sample prediction. Quarters in which pain was more knee-dominant than a person's average predicted lower subsequent sleep quality, independent of overall pain severity. Although population-level sleep-to-pain coupling was not credible, we tested—under a fully theory-driven approach with a priori brain regions—whether individual neuroimaging-based predispositions could still be observed. A previous study identified six brain regions where pain-evoked activation was associated with sleep-deprivation-induced next-day pain amplification. Hypothesizing that baseline activation in these regions may index predisposing traits, we tested whether they moderated sleep-to-pain coupling using baseline pain-evoked fMRI (N=174). The left nucleus accumbens (NAcc) credibly moderated sleep-to-pain coupling, with credible coupling only in individuals with weaker or negative left NAcc responses. All six regions showed moderation in the predicted directions—a pattern unlikely by chance. Pain-evoked anterior cingulate cortex (ACC) activation—tested separately based on preclinical evidence that ACC and NAcc are parallel dopaminergic-gated nodes—also credibly moderated sleep-to-pain coupling in the same direction. In the pain-to-sleep direction, a test of five preclinically identified pain-arousal relay nodes as moderators (fMRI and grey matter volume) yielded no credible effect. Taken together, pain-to-sleep coupling dominates quarterly—reversing typically reported day-to-day asymmetry—and baseline pain-evoked activity in left NAcc and ACC identify individuals whose sleep quality has downstream consequences for pain.

# Keywords

Knee-body pain contrast, Pain arousal pathway, fMRI activation to knee pain, Within-person VARX(1) with coupled innovations.

# Introduction

Sleep disturbance and pain frequently co-occur, with 50–88% of chronic pain patients reporting significant sleep complaints (1, 2). A review of the sleep–pain literature indicates that daily diary studies—in which participants rate their pain and sleep over consecutive days—support a bidirectional relationship, but with a notable asymmetry favoring the sleep-to-pain direction (2). In fact, several studies have reported that daytime pain does not reliably predict next-night sleep quality (3–6), with one exception even favoring the pain-to-sleep direction (7). This sleep-to-pain asymmetry extends to chronic musculoskeletal pain cohorts, as confirmed by a systematic review of 11 daily diary studies (N = 1,014) (8).

Although daily diary studies capture a tight sleep–pain link, they do so over short time periods (days to weeks). Whether the bidirectional coupling observed in micro-longitudinal studies persists at longer intervals (i.e., in panel studies) with the same directional asymmetry remains largely unexplored, with the notable exception of a monthly cross-lagged study of insomnia and self-reported pain over 3 months in temporomandibular disorder (9). At longer timescales (e.g., quarterly), slower processes—inflammatory flare-and-remission cycles, biopsychosocial modulation of pain, changes in habitual sleep behavior, treatment-seeking, and seasonal shifts in activity—can perturb pain and sleep from one assessment to the next, warranting examination of whether the sleep-to-pain asymmetry observed at daily timescales is preserved.

Addressing this question demands appropriate methodology. Decomposition into person-mean and within-person deviation components is common in the related literature (3, 4, 6, 7, 10–16) but not universal (2, 9, 17, 18); at longer timescales, where between-person variation is amplified, omitting this step risks conflating trait-level confounds with temporal coupling (19). Most studies also model each coupling direction in a separate equation, leaving the cross-lagged coefficients unconditioned on each other and the contemporaneous covariance between innovations unmodeled. The two studies that estimate both directions simultaneously (9, 18) address the former but lack the within–between separation, and only the SEM-based approach (18) includes correlated residuals by default. Finally, no study estimates person-specific coupling slopes, precluding identification of characteristics that predispose stronger or weaker coupling (possible moderators). A bivariate multilevel autoregressive model with within-person centering and random slopes addresses all four requirements.

Beyond temporal dynamics, the nature of the reported pain may matter. Individuals whose pain is predominantly localized may experience different sleep consequences than those reporting more widespread presentations, yet to our knowledge, little is known about how within-person shifts in pain location relate to sleep-pain coupling in prospective studies. Knee pain is well suited for examining this question because its localized nature, combined with parallel longitudinal ratings of body pain, allows tracking of within-person shifts between localized and widespread pain phenotypes and testing whether these shifts moderate coupling strength.

Neural mechanisms may also gate the sleep-pain relationship. In the sleep-to-pain direction, sleep disruption is thought to blunt endogenous pain inhibition and increase pain sensitivity through central mechanisms (20–22). Krause et al., (2019) showed that acute sleep deprivation amplifies pain reactivity within primary somatosensory cortex (S1) yet blunts reactivity in higher-order valuation regions—the middle insula, thalamus, anterior insula, and bilateral nucleus accumbens (NAcc) (23). The magnitude of increased S1 reactivity and the degree of decreased thalamic reactivity each predicted the lowering of pain thresholds across individuals (23). However, they acknowledged that inferences regarding the mechanisms of reduced thresholds rely on correlations with suprathreshold neural pain reactivity, not formal mediation tests, leaving open whether these activation changes reflect causal mechanisms or trait-level vulnerability markers that identify individuals predisposed to stronger sleep-to-pain coupling over time—in which case, individual differences in pain-evoked activation measured at a single time point could serve as moderators of that coupling.

Among these regions, the NAcc is of particular interest given its role in reward circuitry, where altered function has been linked to pain chronification and disrupted endogenous pain modulation (24–27). Preclinical work further implicates dopaminergic mechanisms: Sardi et al. (2024) showed that sleep restriction in rats decreases dopamine levels in the ACC and NAcc, and that pharmacological activation of D2 receptors in either region prevents sleep-restriction-induced hyperalgesia (28), consistent with human findings (29). The parallel role of the ACC and NAcc as D2-gated nodes motivates testing whether pain-evoked ACC activation—beyond the five Krause et al. (2019) regions that include the NAcc—also predisposes individuals’ sleep-to-pain coupling.

In the pain-to-sleep direction, preclinical work has identified circuit-level mechanisms through which pain disrupts sleep, including hypothalamic dynorphin/kappa opioid receptor (KOR) signaling that promotes wakefulness under chronic neuropathic pain (30). Lynch et al. (2025) further identified a pontine-to-forebrain arousal pathway in which calcitonin gene-related peptide (CGRP)-expressing neurons in the parabrachial nucleus (PBN) relay nociceptive signals to four wake-promoting forebrain targets (31): he substantia innominata of the basal forebrain (SI-BF/Ch4), central nucleus of the amygdala (CeA), bed nucleus of the stria terminalis (BNST)—the latter two forming part of the extended amygdala (32)—and lateral hypothalamus (LH). If these circuits are conserved in humans, individual differences in the structural or functional properties of these relay nodes could predispose individuals to stronger pain-to-sleep coupling.

To address these questions, the present study leveraged a longitudinal cohort of adults (N = 229) who completed quarterly numerical rating scales of knee pain, body pain, and sleep quality over up to 11 time points, with a subset (N = 174) completing task-based fMRI during painful knee stimulation at baseline. The study had **four aims**. First, using a Bayesian bivariate autoregressive (VARX(1)) model fitted to within-person deviations, with correlated innovations and person-specific slopes, we tested whether the sleep-to-pain asymmetry observed in daily studies holds at the quarterly timescale. Second, we tested whether within-person shifts in the knee-versus-body pain contrast moderated coupling strength. Third, we tested whether pain-evoked BOLD activation at baseline in the Krause et al. (2019) regions (23)— with the NAcc analyzed separately by hemisphere—and the ACC (28) moderated sleep-to-pain coupling. Fourth, we tested whether five nodes of the Lynch et al. (2025) pain-arousal relay pathway moderated pain-to-sleep coupling, using pain-evoked BOLD activation and grey matter volume as putative proxies for arousal relay capacity. Given the modest neuroimaging sample, all regions were selected a priori from published frameworks rather than through data-driven exploration; to our knowledge, these are the only frameworks linking specific brain regions to sleep–pain interactions.

# Methods

## Notation

**Table 1.** Notation glossary.

| Symbol       |                           Meaning                           |
|:-------------|:-----------------------------------------------------------:|
| $P$, $S$ |                    Pain and sleep scores                    |
| $K$           |          Pain localization (knee-vs-body contrast)          |
| $X$           |                   Neuroimaging moderator                    |
| $W$           |   Generic conditioning variable (Johnson-Neyman analysis)   |
| $\mu$           |                          Intercept                          |
| $\varphi$           |                       Autoregression                        |
| $\lambda$           |                    Cross-lagged coupling                    |
| $\delta$           |             Direct effect of pain localization              |
| $\omega$           | Within-person interaction (coupling$\times$ localization) |
| $\gamma$           |                  Between-person moderation                  |
| $\tau$           |                      Random effect SD                       |
| $\sigma$           |                        Innovation SD                        |
| $\rho$           |                   Innovation correlation                    |
| $u$           |          Person-specific random coupling deviation          |
| $\varepsilon$           |                         Innovation                          |

**Note.** Uppercase letters denote observed variables and scores; lowercase Greek letters denote model coefficients and parameters. Subscripts: $p$ = pain, $s$ = sleep, $sp$ = sleep-to-pain direction, $ps$ = pain-to-sleep direction. Superscript $w$ = within-person centered.

## Participants

Participants were drawn from a multisite longitudinal observational study examining ethnic/race group differences in individuals with or at risk for knee OA, conducted at the University of Florida (UF) and the University of Alabama at Birmingham (UAB). Some of the methods and results related to this larger project can be found elsewhere (33–43). The study was approved by the UF and UAB Institutional Review Boards, with UF serving as the IRB of Record providing regulatory and ethical oversight. All participants provided verbal and written informed consent, and both studies were carried out in accordance with the Declaration of Helsinki. Individuals between 45 and 85 years of age were screened for symptomatic knee OA. Full exclusion criteria have been previously reported in the abovementioned papers; briefly, individuals were excluded for clinically significant surgery to the index (most painful) knee, uncontrolled hypertension, heart failure or history of acute myocardial infarction, peripheral neuropathy, systemic rheumatoid disorders, daily opioid use, cognitive impairment, psychiatric illness, neurological disease, significantly greater pain in a body region other than the knee, or pregnancy/nursing. Pain and sleep assessments were collected quarterly for up to 11 time points. A subset of participants also completed task-based fMRI during painful knee stimulation at the baseline session.

## Measures

### Quarterly assessments

At each quarterly visit, participants completed a set of self-report items assessing knee pain, body pain, and sleep quality. Two binary gateway items asked whether the participant had experienced any knee pain (q1) or any pain in the body (q6) in the past 7 days. These were followed by nine 0–10 numerical rating scale items (**Table 2**): four for knee pain intensity and stiffness (q2–q5), four for body pain intensity and stiffness (q7–q10), and one for sleep quality (q13). Additional items assessed weekly fatigue (q11; 0 = no fatigue, 10 = as much fatigue as you can imagine) and mood (q12; 0 = very negative, 10 = very positive); these items were not included in the present analyses. When a participant reported no knee pain (q1 = 0) or no body pain (q6 = 0), missing intensity ratings for the corresponding region were set to zero. The gateway items themselves did not enter any statistical model.

**Table 2.** Quarterly assessment items.

| Item | Description | Scale |
|:---|:--:|:--:|
| q2 | Worst knee pain in the past 7 days | 0 (no pain) – 10 (worst imaginable) |
| q3 | Average knee pain in the past 7 days | 0–10 |
| q4 | Current knee pain | 0–10 |
| q5 | Knee stiffness in the past 7 days | 0 (no stiffness) – 10 (worst imaginable) |
| q7 | Worst body pain in the past 7 days | 0 (no pain) – 10 (worst imaginable) |
| q8 | Average body pain in the past 7 days | 0–10 |
| q9 | Current body pain | 0–10 |
| q10 | Body stiffness in the past 7 days | 0 (no stiffness) – 10 (worst imaginable) |
| q13 | Sleep quality in the past 7 days | 0 (very poorly) – 10 (very well) |

**Note.** Higher pain scores = more pain; higher sleep score (q13) = better sleep quality.

### Factor analysis and imputations

Because the eight pain items span two body regions (knee, body), they were expected to reflect at least two dimensions: overall pain severity and a knee-versus-body contrast. To recover these dimensions, the eight items (q2–q5, q7–q10) were submitted to exploratory factor analysis (EFA) with polychoric correlations and varimax rotation. Retained factors based on Horn's parallel analysis (44), which compares observed eigenvalues against those obtained from random data of the same dimensions (1,000 replications) to distinguish substantive factors from sampling noise. We supplemented interpretability of the factor describing the knee-versus-body contrast based on clinical correlates (described in the next section). Bartlett factor scores were computed for each person-timepoint, requiring a minimum of two of the eight pain items; when fewer than eight items were available, factor scores were estimated from the available subset using the corresponding rows and columns of the factor loading and covariance matrices. The sleep item (q13) was z-standardized across all person-timepoints to place it on the same scale as the Bartlett factor scores, which are in z-score units by construction.

Single interior missing quarters in any of the three scores (pain intensity, knee–body contrast, and z-scored sleep quality) were recovered by linear interpolation between the two flanking observed values, provided both neighbors were present. Longitudinal records were then partitioned into maximal runs of consecutive quarters in which both pain intensity and sleep quality scores were non-missing, and any run shorter than three quarters was discarded. Observations were further excluded if any of the three lagged scores was missing at t−1, as all three enter the bivariate VARX model as predictors.

### Baseline clinical pain measures

At the baseline visit, participants completed several validated assessments of knee pain severity and function. These measures serve to characterize the sample clinically and to provide external validation of the quarterly pain measures derived from the factor analysis described above. The Western Ontario and McMaster Universities Osteoarthritis Index (WOMAC) (45) assessed knee-specific pain (5 items: pain during walking, stairs, at night in bed, sitting/lying, and standing; each 0 = none to 4 = extreme; sum), stiffness (2 items; sum), and physical function (17 items; sum), as well as a total score (sum of all 24 items). The Pain History Questionnaire (PHQ) assessed the number of days per week of knee pain, the percentage of the waking day spent in knee pain, duration of knee pain, and pain areas endorsed across 13 body regions (hands, arms, shoulders, neck, head/face/jaw, chest, stomach, pelvis, upper back, lower back, knees, legs, and feet/ankles). Kellgren-Lawrence (KL) grades for the index knee were obtained from weight-bearing anteroposterior radiographs, scored by a musculoskeletal radiologist on a 0–4 scale (0 = no OA features, 4 = severe) (46).

### MRI dataset

A subset of 182 participants completed task-based fMRI, had a corresponding structural T1 at the baseline visit, and contributed quarterly data to the coupling analyses. The stimulation paradigm consisted of five 24-second blocks of mechanical pain stimulation alternating with 24-second rest periods over approximately 6 minutes (150 volumes). Painful stimulation was delivered using a von Frey filament applied to the participant's most painful knee (right in 98 participants, left in 84). Each 72-second cycle comprised a rest period, a stimulation period, and a post-stimulation rest period separating consecutive stimulation blocks.

Imaging data were acquired at two sites using Philips Achieva 3T scanners: the University of Florida McKnight Brain Institute (UF; software version 3.2.1; 116 participants) and the University of Alabama at Birmingham (UAB; software version 3.2.3; 72 participants). T1-weighted structural images were acquired using a 3D MPRAGE sequence (1 mm isotropic voxels, flip angle = 8°) with nearly identical parameters across sites. Functional images were acquired using a 2D gradient-echo echo-planar imaging (EPI) sequence (TR = 2.4 s, TE = 30 ms, flip angle = 90°, slice thickness = 3.5 mm, in-plane voxel size = 2.9 $\times$ 2.9 mm, 150 volumes). Acquisition parameters were matched across sites, with minor differences in phase encoding steps (UF: 72; UAB: 64) and echo train length (UF: 39; UAB: 35); the reconstructed matrix size was identical (80 $\times$ 80).

### MRI preprocessing and first-level analysis

All image processing was performed using SPM12 (Wellcome Centre for Human Neuroimaging, London, UK). T1-weighted structural images were segmented into grey matter (GM), white matter (WM), and cerebrospinal fluid (CSF) tissue classes using unified segmentation (47) with SPM's default tissue probability maps, bias regularization of 0.001, 60 mm bias FWHM, and sampling distance of 3 mm. DARTEL-imported tissue maps were generated for GM, WM, and CSF. A study-specific anatomical template was then created from the DARTEL-imported tissue maps (rc1, rc2, rc3) of all participants using the Diffeomorphic Anatomical Registration Through Exponentiated Lie Algebra (DARTEL) algorithm (48). The DARTEL template estimation proceeded through six outer iterations (regularization parameters: [4 2 1e-6], [2 1 1e-6], [1 0.5 1e-6], [0.5 0.25 1e-6], [0.25 0.125 1e-6], [0.25 0.125 1e-6]; time steps $K$: 0, 0, 1, 2, 4, 6) with three inner iterations each, producing subject-specific flow fields encoding the deformation from native to template space. The final DARTEL template was registered to MNI space via an affine transformation (48). Modulated normalized GM images (smwc1; 1.5 mm isotropic) were generated from this pipeline for the VBM-based ROI analyses described below.

Functional images were preprocessed in the following order: (1) slice timing correction using a Philips-specific interleaved acquisition order (groups of $round(\sqrt{N_{slices}})$ slices) with the middle slice as reference; (2) motion correction via realignment and unwarping, with quality factor = 1.0, separation = 3 mm, registration smoothing = 5 mm FWHM, 7th-degree B-spline interpolation, and first-order effects of pitch and roll modeled for susceptibility-by-movement interactions; (3) rigid-body coregistration of the mean functional image to the subject's T1-weighted structural image using normalized mutual information (separation = [4 2] mm, smoothing = [7 7] mm FWHM), with the transformation applied to all functional volumes. Since DARTEL delivers large deformations, we used the push-forward warping method to preserve all data from the native functional images. The T1-derived DARTEL flow fields were applied to the coregistered functional images, resampling to 3 $\times$ 3 $\times$ 3 mm isotropic voxels without Jacobian modulation (preserving signal concentration). This method naturally furnishes both spatially non-smoothed and smoothed normalized images; the smoothed versions were generated with a 6 mm full-width at half-maximum (FWHM) Gaussian kernel.

For each participant, a general linear model (GLM) was specified in SPM12 with the five stimulation blocks (onsets: 24, 96, 168, 240, and 312 s; duration: 24 s each) convolved with the canonical hemodynamic response function as the sole task regressor. Nuisance regressors included the average white matter BOLD signal, the average cerebrospinal fluid BOLD signal, and six rigid-body motion parameters (three translations, three rotations) from the realignment step. A discrete cosine transform high-pass filter with a cutoff of 1/128 Hz was applied to remove low-frequency drift from both the data and the design matrix. The contrast of interest was the T-contrast for the effect of stimulation versus implicit baseline (rest and post-stimulation periods combined).

### Sleep-to-pain ROIs

All ROIs in this and the following section were defined a priori from published frameworks, with no data-driven selection. Within the Krause et al. (2019) sleep deprivation framework, six spherical ROIs were defined at coordinates where pain-evoked activation changed following one night of total sleep deprivation (23). The original regions comprised right S1, right middle insula, left thalamus, left anterior insula, and bilateral NAcc—hemisphere assignments that presumably reflected either contralateral somatotopic reactivity to the stimulation site (left calf) or lateral specialization unrelated to primary nociceptive mapping. We retained Krause et al.'s coordinates and ROI sizes but adapted hemisphere assignments for somatotopically organized regions to match the stimulated side, which varied across our sample (see **Figure S4**).

- Given the well-established contralateral dominance of S1 during nociceptive stimulation (23, 49), this ROI was extracted from the hemisphere contralateral to the stimulated knee (MNI ±36, −45, 59; radius = 8 mm; 82 voxels).

- The middle insula, which similarly exhibits a contralateral bias (23), was treated the same way (MNI ±32, 4, 11; radius = 8 mm; 72 voxels).

The remaining ROIs were considered to reflect lateral specialization rather than stimulus-side dependence and were therefore retained at their original Krause et al. (2019) coordinates.

- The thalamus was activated ipsilaterally in Krause et al. (2019) (left; MNI −10, −6, 10; radius = 4 mm; 10 voxels), arguing against somatotopic reactivity—and while some contralateral thalamic bias has been reported, it is modest (~20% greater contralateral than ipsilateral activation, compared to ~110% for S1 (49)).

- The anterior insula was retained in the left hemisphere (MNI −27, 25, 0; radius = 8 mm; 82 voxels), consistent with evidence of bilateral-to-left-lateralized insular pain responses (49).

- Both nucleus accumbens ROIs (NAcc; MNI ±9, 2, −7; radius = 6 mm; 17 voxels) were kept as bilateral fixed-coordinate regions, as they index dopamine-mediated pain inhibition rather than somatotopic nociceptive processing.

An additional spherical ROI was defined in the right dorsal ACC/mid-cingulate cortex (dACC/MCC) at the peak coordinate from a meta-analysis of 222 pain fMRI experiments (50) (MNI 6, 12, 38; radius = 6 mm; 32 voxels; Figure S4), consistent with right-lateralized ACC pain responses (51). This ROI was motivated by Sardi et al. (2024), who demonstrated that the ACC and NAcc operate as parallel D2-gated nodes: D2 agonist infusion into either region prevents sleep-restriction-induced hyperalgesia in rats (28). Notably, the meta-analytic peak also overlaps with the human analogue of the rat dACC (Cg1) region (52) targeted by Sardi et al. Because this ROI was defined on the basis of pain processing rather than derived directly from Sardi et al.'s stimulation sites, a left-hemisphere counterpart was also included by mirroring the x-coordinate (MNI −6, 12, 38; radius = 6 mm; 32 voxels; **Figure S4**).

### Pain-to-sleep ROIs

Within the Lynch et al. (2025) pain-arousal relay framework (31), five bilateral ROIs corresponding to nodes of the ${PBN}^{elCGRP}$ pain-arousal pathway were defined using published probabilistic atlases to maximize anatomical specificity for these small subcortical structures (**Figure S6**).

- The lateral parabrachial nucleus (PBN) was defined from the Brainstem Navigator atlas (53), which provides probabilistic parcellations of brainstem nuclei in MNI152 space at 1 mm isotropic resolution; labels 19 and 20 (left and right lateral PBN) were combined into a single bilateral mask (3 brain voxels at 3 mm fMRI resolution).

- The substantia innominata/basal forebrain Ch4 cell group (SI-BF/Ch4) was defined from the probabilistic cytoarchitectonic atlas of Zaborszky et al. (2008), which provides continuous probability maps in MNI152 space (54); the Ch4 probability map was thresholded at any probability \> 0 and used as a weighted extraction mask.

- The central nucleus of the amygdala (CeA) was defined from the CIT168 high-resolution in vivo atlas (55); because the published atlas provides an extended amygdala volume (combining CeA and BNST), we constructed a CeA-specific probabilistic map by extracting the centromedial amygdala label (AMY_CEN, label 4) from the crowd-sourced individual-observer labelings available in the atlas repository (2 observers $\times$ 8 template brains = 16 labelings), averaging the binary masks into a probability map in CIT168 native space (0.7 mm isotropic), registering to MNI152 1 mm space via FSL FLIRT (12-parameter affine) followed by FNIRT (nonlinear) because the CIT168 atlas is defined in its own template space rather than MNI152, and mirroring across the midline to create a bilateral atlas.

- The bed nucleus of the stria terminalis (BNST) was defined from the probabilistic atlas of Theiss et al. (2017), derived from 3T MRI data in MNI152 space (56).

- The lateral hypothalamus (LH) was defined from the probabilistic hypothalamic atlas (57), i.e., labels 25 (left) and 26 (right).

All five ROIs were extracted bilaterally regardless of the stimulated knee, because these structures belong to the spino-parabrachio-amygdaloid and medial pain pathways, which receive predominantly bilateral projections and may lack the somatotopic organization characteristic of the lateral pain system.

### ROI image data extraction

For each ROI, mean contrast values were extracted from the first-level stimulation-versus-baseline contrast images (con_0001.nii). For spherical ROIs, the unweighted mean was computed across all voxels within the sphere. For atlas-defined probabilistic ROIs, probability-weighted means were computed: ${\overline{X}}_{i} = \sum_{v}{BOLD}_{v} \cdot w_{v}\,/\,\sum_{v}w_{v}$, where $w_{v}$ is the atlas probability at voxel $v$. Atlases were resampled to the fMRI resolution (3 mm isotropic) using linear interpolation prior to extraction. For each atlas-defined ROI, probability-weighted grey matter volume was also computed from modulated GM images (smwc1 output from SPM DARTEL normalization, 1.5 mm isotropic, $V_{vox} = {1.5}^{3} = 3.375$ mm³): ${Vol}_{i} = \sum_{v}{GM}_{v} \cdot w_{v} \cdot V_{vox}$, where ${GM}_{v}$ is the modulated grey matter probability at voxel $v$. All five atlas-defined ROIs were viable at VBM resolution, including the PBN which was largely unreliable with only 3 voxels at the coarser fMRI resolution.

A gray matter mask derived from a study-specific DARTEL template (GM probability \> 0.25) was applied to restrict gray matter ROIs. However, due to unacceptable voxel dropout in structures spanning gray and white matter (e.g., thalamus), partial volume effects (e.g., ACC), insufficient size or inferior location to survive masking at 3 mm resolution (e.g., PBN), or mixed tissue composition (e.g., insula, basal forebrain), the mask was applied only to the NAcc and LH ROIs.

## Statistical model

### Within-person decomposition

All time-varying variables ($Y \in P,S,K$) were decomposed into between-person means (${\overline{Y}}_{i}$) and within-person deviations ($Y_{it}^{w} = Y_{it} - {\overline{Y}}_{i}$). Only within-person components entered the model, removing all stable between-person confounds by construction (58), a critical step given that standard cross-lagged panel models conflate within- and between-person variance and can yield misleading estimates of temporal dynamics (19). Lagged within-person variables ($Y_{i,t - 1}^{w}$) were created within continuous observation segments. Interaction terms ($S \times K$, $P \times K$) were computed from lagged within-person components; because these components are person-mean centered by construction, their products are correctly centered without further adjustment.

### Bivariate VARX(1) specification

To estimate the bidirectional within-person coupling between pain and sleep while accounting for autoregressive dynamics and pain localization, a Bayesian bivariate vector autoregressive model with exogenous contrast moderation (VARX(1)) was specified:

$P_{it}^{w}=\mu_{p}+\varphi_{p}P_{i,t-1}^{w}+\lambda_{sp,it}S_{i,t-1}^{w}+\delta_{p}K_{i,t-1}^{w}+\varepsilon_{p,it}$

$S_{it}^{w}=\mu_{s}+\lambda_{ps,it}P_{i,t-1}^{w}+\varphi_{s}S_{i,t-1}^{w}+\delta_{s}K_{i,t-1}^{w}+\varepsilon_{s,it}$

where the coupling coefficients $\lambda_{sp,it}$ and $\lambda_{ps,it}$ are allowed to vary as a function of demographic covariates, within-person pain localization contrast ($K^{w}$; knee-vs-body; see **Table 1**), and person-specific random effects:

$\lambda_{sp,it}=\lambda_{sp}+\gamma_{sp,age}\,{Age}_{i}^{z}+\gamma_{sp,sex}\,{Sex}_{i}^{c}+\omega_{sp}\,K_{i,t-1}^{w}+u_{sp,i}$

$\lambda_{ps,it}=\lambda_{ps}+\gamma_{ps,age}\,{Age}_{i}^{z}+\gamma_{ps,sex}\,{Sex}_{i}^{c}+\omega_{ps}\,K_{i,t-1}^{w}+u_{ps,i}$

Here $\lambda_{sp}$ and $\lambda_{ps}$ are the population-average coupling coefficients, $\gamma_{age}$ and $\gamma_{sex}$ adjust coupling for demographic differences (included as nuisance covariates so that $\lambda_{sp}$ and $\lambda_{ps}$ represent coupling at sample-average age and sex), $\omega$ captures within-person moderation by pain localization, and $u_{sp,i} \sim N(0,\tau_{sp}^{2})$ and $u_{ps,i} \sim N(0,\tau_{ps}^{2})$ are person-specific random deviations. Age was z-scored; sex was coded 0 = male, 1 = female and centered at the sample mean (${Sex}_{i}^{c} = {Sex}_{i} - 0.646$). The within-person pain localization contrast ($K^{w}$) also entered as a direct effect on each dependent variable ($\delta_{p}$, $\delta_{s}$).

Given the definition of these measures—higher values indicate more pain but better sleep quality—the intuitive direction of coupling is negative for both pathways: more pain should predict worse (lower) subsequent sleep quality ($\lambda_{ps} < 0$), and better (higher) sleep quality should predict less (lower) subsequent pain ($\lambda_{sp} < 0$). Positive coupling would reflect seemingly counterintuitive but plausible dynamics (e.g., more pain predicting better sleep). Accordingly, throughout this paper, "stronger coupling" denotes a more negative coefficient (larger in absolute value), and our intuition was that any moderator that strengthens coupling pushes it further below zero. Thus, throughout this paper, besides the 95% equal-tailed credible intervals (CrI) for all parameters of the model (ARs, direct effects of $K^{w}$, and moderation coefficients), one-sided posterior probabilities $P( < 0)$ are reported for coupling coefficients.

Innovations were modeled as correlated using a Cholesky decomposition: $\varepsilon_{p,it} \sim N(0,\sigma_{p}^{2})$ and $\varepsilon_{s,it}|\varepsilon_{p,it} \sim N(\rho\sigma_{s}/\sigma_{p} \cdot \varepsilon_{p,it},\sigma_{s}^{2}(1 - \rho^{2}))$, equivalent to bivariate normal innovations with correlation $\rho$. Weakly informative priors were placed on all parameters: fixed effects $\sim N(0,5)$; moderation parameters $\gamma \sim N(0,1)$; random effect SDs $\tau\sim HalfCauchy(1)$; innovation SDs $\sigma\sim HalfCauchy(2)$; and innovation correlation $\rho\sim Beta(2,2)$ rescaled to $\lbrack - 1,\ 1\rbrack$.

Estimation used the No-U-Turn Sampler (NUTS) with 4 chains, 2,000 tuning iterations, and 2,000 posterior draws per chain (8,000 total). All models were implemented in PyMC 5.27.1 (59), a probabilistic programming library for Python that provides tools for constructing and fitting Bayesian models using intuitive syntax and supporting a wide range of statistical models with GPU acceleration via PyTensor backends, leveraging the GPU resources available on the University of Florida's HiPerGator high-performance computing cluster.

### Model comparison

To assess the evidence for each coupling direction, four nested models were fit and compared via approximate leave-one-out cross-validation (LOO-CV) using Pareto-smoothed importance sampling (PSIS) (60), as implemented in ArviZ (61): (1) the full model with both coupling directions; (2) pain-to-sleep coupling only ($\lambda_{sp} = 0$, no $u_{sp}$); (3) sleep-to-pain coupling only ($\lambda_{ps} = 0$, no $u_{ps}$); and (4) no coupling ($\lambda_{sp} = \lambda_{ps} = 0$). All models retained coupled innovations, contrast terms, and age/sex moderation on whichever coupling directions were present. LOO-CV estimates how well each model would predict new, unseen observations by approximating the result of repeatedly refitting the model with one observation left out, without actually performing the computationally expensive refitting. The comparison metric is the expected log pointwise predictive density (elpd), which summarizes each model’s out-of-sample predictive accuracy: higher elpd values indicate better prediction, and differences between models ($\Delta$elpd) quantify the predictive contribution of each coupling pathway. The reliability of the LOO-CV approximation was assessed via Pareto $\widehat{k}$ diagnostics, which flag individual observations where the importance sampling approximation may be inaccurate ($\widehat{k} > 0.7$).

### Neuroimaging moderator analyses

For each moderator analysis, the ROI's z-scored value ($X_{i}$; fMRI response contrast or GM volume) was added as an additional between-person term in the coupling coefficients:

$\lambda_{sp,it}=\lambda_{sp}+\gamma_{sp,age}{Age}_{i}^{z}+\gamma_{sp,sex}{Sex}_{i}^{c}+\omega_{sp}K_{i,t-1}^{w}+\gamma_{sp}X_{i}+u_{sp,i}$

(and analogously for $\lambda_{ps,it}$). The parameters $\gamma_{sp}$ and $\gamma_{ps}$ quantify how a one-SD increase in the moderator shifts the population coupling slope. For the Krause et al. (2019) ROIs (23) and the ACC (Sardi et al. (2024) framework (28)), we tested moderation of the sleep-to-pain coupling ($\gamma_{sp}$) because these theoretical frameworks concern how sleep loss amplifies pain. For the Lynch et al. (2025) arousal pathway ROIs (31), we tested moderation of the pain-to-sleep coupling ($\gamma_{ps}$) because the theoretical framework concerns how pain disrupts sleep via arousal relay circuits. Each ROI was tested as a moderator in a separate model run.

Because sleep deprivation increased pain-evoked activation in S1 but decreased it in the middle insula, thalamus, anterior insula, and NAcc in Krause et al.’s study (23), and because stronger coupling corresponds to a more negative coefficient (see Measures), we expected negative $\gamma_{sp}$ for S1 and positive $\gamma_{sp}$ for the remaining ROIs. For the ACC, Sardi et al. (2024) showed that it operates as a parallel D2-gated node alongside NAcc (28), so we also expected a positive $\gamma_{sp}$ as with the NAcc. For the Lynch et al. (2025) arousal pathway ROIs (31), greater relay capacity (higher pain-evoked BOLD or larger grey matter volume) was expected to strengthen pain-to-sleep coupling (negative $\gamma_{ps}$). Although directional predictions are specified for both frameworks, whether these translate from acute or preclinical designs to trait-level moderation of longitudinal coupling remains uncertain. Thus, we still report the 95% equal-tailed CrI for all moderation parameters, with sign expectations serving as interpretive benchmarks rather than formal directional hypotheses.

### Johnson-Neyman analysis

To characterize the continuous range of moderator values over which coupling was credibly different from zero, Johnson-Neyman (JN) analyses were conducted for all moderation effects. For a given conditioning value $W$, the conditional coupling slope is:

$\widehat{\lambda}(W)={\widehat{\lambda}}_{0}+\widehat{\theta}W$

where ${\widehat{\lambda}}_{0}$ is the population-average coupling coefficient (equivalent to the adjusted intercept in a frequentist regression framework; ${\widehat{\lambda}}_{sp}$ for sleep-to-pain, ${\widehat{\lambda}}_{ps}$ for pain-to-sleep) and $\widehat{\theta}$ is the corresponding moderation parameter (${\widehat{\omega}}_{sp}$ or ${\widehat{\omega}}_{ps}$ for the within-person contrast moderator; ${\widehat{\gamma}}_{sp}$ or ${\widehat{\gamma}}_{ps}$ for between-person neuroimaging moderators). Here $W$ is a free conditioning variable—not a specific person's or quarter's value, but a hypothetical moderator value at which the population-average coupling is evaluated. Because both ${\widehat{\lambda}}_{0}$ and $\widehat{\theta}$ are estimated as full posterior distributions, the posterior of $\widehat{\lambda}(W)$ was computed draw-by-draw for a dense grid of $W$ values spanning the observed moderator range. At each $W$, the 2.5th and 97.5th percentiles of the posterior defined the 95% credible interval (equal-tailed CrI). The JN boundary was identified as the moderator value at which the credible interval first included zero. That is, the threshold beyond which the conditional coupling is no longer credibly different from zero.

For the within-person contrast moderator ($K^{w}$), $W$ ranges over the observed within-person centered pain localization values pooled across all persons and quarters. The JN boundary identifies a quarter-level threshold: it is the pain localization value at which the population-average coupling transitions from credibly nonzero to non-credible. Because pain localization varies within persons across quarters, the same individual may fall in the credible region in some quarters and not in others, depending on their pain pattern that quarter. For between-person moderators (i.e., $X_{i}$), the JN boundary identifies a person-level threshold: individuals whose moderator value falls within the credible region exhibit coupling that is credibly different from zero.

In the classical frequentist framework, the JN boundary is obtained by solving a quadratic equation for the moderator value at which the $t$-statistic equals the critical value, yielding a closed-form solution that depends on the variance-covariance matrix of the estimated coefficients (62). In the Bayesian framework, no single standard error or test statistic exists; instead, the full joint posterior of $({\widehat{\lambda}}_{0},\widehat{\theta})$ determines the credible interval at each $W$. Because the joint posterior is not constrained to be Gaussian—it may be skewed, heavy-tailed, or otherwise non-normal—the frequentist quadratic solution does not apply, and hypothesis tests based on a $t$-distribution are not available. The grid-based approach used here is the Bayesian analogue: it accommodates any posterior shape and is computationally trivial. Specifically, 500 equally spaced grid points spanning the observed moderator range were evaluated across all 8,000 posterior draws. The JN boundary was then refined by linear interpolation between the two adjacent grid points where the credible interval bound crossed zero, yielding approximate sub-grid precision.

# Results

## Factor analysis and final sample

An exploratory factor analysis was conducted to separate general pain severity from the knee-versus-body pain distribution. The first factor was dominant (eigenvalue = 5.51, 68.9% of variance) and exceeded the 95th-percentile random eigenvalue threshold (1.12). The second eigenvalue (1.45, 18.1% of variance) also exceeded its parallel analysis threshold (1.08), indicating that both factors are retained on statistical as well as theoretical grounds. The retained two-factor solution jointly accounted for 87.0% of the variance. The first factor (F1: General Pain) had all eight items loading positively (range: 0.78–0.86), capturing overall pain severity. The second factor (F2: Contrast) showed positive loadings on knee items (+0.31 to +0.45) and negative loadings on body items (-0.27 to -0.50), capturing the within-person contrast between knee-localized and body-wide pain. The two factors were orthogonal ($r = 0.018$).

Of the 243 participants in the parent study, 229 had at least one continuous segment of three or more consecutive quarterly timepoints with both pain factor scores and sleep scores available, while 14 were excluded for lacking such a segment (**Figure 1**). Among retained participants, the median number of usable lag-1 transitions was 9 (range: 2–10), totaling 1,818 transitions. **Table 3** presents the demographic and baseline clinical characteristics of this final analytic sample.

![](../results/step03_varx_data/step03_figure1.png)

**Figure 1.** Data availability grid (participants $\times$ quarters). Blue dots indicate observed retained data points; red dots indicate retained points for which scores were interpolated (118 of 2,056 retained points); grey dots indicate observations discarded due to segment length < 3. Horizontal lines connect consecutive quarters within retained segments. Participants are sorted by number of retained points (bottom = fewest). Participants below the dashed line (N = 14) were excluded for lacking any segment of three or more consecutive quarters.

**Table 3.** Demographic and baseline clinical characteristics (N = 229).

| Variable | Level | Value |
|:---|:--:|:--:|
| Age, years, mean (SD) [range] |  | 58.1 (8.2) [44–80] |
| Female sex, N (%) |  | 148 (64.6) |
| Race/ethnicity, N (%) |  |  |
|  | Black/African American | 91 (39.7) |
|  | White/Caucasian | 67 (29.3) |
|  | Hispanic/Latino | 55 (24.0) |
|  | Other | 16 (7.0) |
| BMI, kg/m², mean (SD) [range] |  | 31.6 (7.7) [18.7–65.4] |
| WOMAC Pain (0–20), mean (SD) |  | 5.8 (5.0) |
| WOMAC Stiffness (0–8), mean (SD) |  | 2.5 (2.2) |
| WOMAC Physical Function (0–68), mean (SD) |  | 18.5 (16.1) |
| WOMAC Total (0–96), mean (SD) |  | 26.7 (22.6) |
| PHQ knee pain days per week, mean (SD) |  | 3.9 (2.8) |
| PHQ % waking day in knee pain, mean (SD) |  | 36.3 (33.4) |
| Knee pain rating (0–100), mean (SD) |  | 11.1 (18.6) |
| PHQ endorses knee pain, N (%) |  | 162 (70.7) |
| Kellgren-Lawrence grade, N (%) |  |  |
|  | 0 | 70 (31.2) |
|  | 1 | 43 (19.2) |
|  | 2 | 48 (21.4) |
|  | 3 | 34 (15.2) |
|  | 4 | 29 (12.9) |

**Note.** WOMAC = Western Ontario and McMaster Universities Osteoarthritis Index. PHQ = Pain History Questionnaire. Knee pain rating: 0–100 numerical rating scale during quantitative sensory testing. Kellgren-Lawrence grades from weight-bearing radiographs of the index knee (5 missing). BMI = body mass index.

Finally, of the 229 participants in the coupling sample, 174 had usable task-based fMRI data from the baseline session and were included in the fMRI moderation analyses (115 female, 59 male; mean age = 58.7 years, SD = 8.5; 71 Black/African American, 47 White/Caucasian, 42 Hispanic/Latino, 14 other).

## External validation of the pain localization contrast

We provided external validation for the contrast factor. Each participant's mean contrast score across all available quarters ($\overline{K}$) was compared against the baseline pain area endorsements and clinical measures. Because the PHQ body map allows endorsement of multiple pain areas simultaneously (mean = 3.2 areas, SD = 2.5), participants were classified into three pain distribution groups: knee pain only (N = 19), knee pain plus at least one other area (N = 143), and no knee pain (N = 67). A one-way ANOVA revealed significant differences in $\overline{K}$ across groups ($F(2,226) = 16.56$, $p < 0.001$). Tukey post-hoc comparisons confirmed that all three groups differed from each other: knee-only participants had the highest average and positive contrast scores ($\overline{K} = 0.57$, SD = 0.64), followed by knee-plus-others ($\overline{K} = 0.10$, SD = 0.72; $p = 0.016$ vs knee-only), and no-knee participants had negative contrast scores ($\overline{K} = - 0.34$, SD = 0.61; $p < 0.001$ vs knee-only and $p < 0.001$ vs knee-plus-others). Complementarily, point-biserial correlations between each of the 13 individual pain area endorsements and $\overline{K}$ confirmed a clear pattern: knee endorsement was positively associated with the contrast ($r_{pb} = 0.31$, $p < 0.001$), while all 12 non-knee areas showed negative or near-zero associations, with upper back ($r_{pb} = - 0.16$, $p = 0.018$) and lower back ($r_{pb} = - 0.14$, $p = 0.034$) showing the strongest negative effects; however, only knee endorsement survived FDR correction across the 13 tests. These results are illustrated in **Figure S1**.

Other baseline clinical measures further confirmed this pattern. The person-mean contrast correlated positively with all knee-specific continuous measures: PHQ knee pain days per week ($r = 0.37$, $p < 0.001$), PHQ percent of waking day in knee pain ($r = 0.30$, $p < 0.001$), WOMAC Pain ($r = 0.28$, $p < 0.001$), WOMAC Total ($r = 0.26$, $p < 0.001$), WOMAC Physical Function ($r = 0.25$, $p < 0.001$), WOMAC Stiffness ($r = 0.24$, $p < 0.001$), knee pain rating ($r = 0.23$, $p < 0.001$), and radiographic OA severity (Kellgren-Lawrence grade of the index knee; Spearman $\rho = 0.31$, $p < 0.001$). Scatter plots for all measures are shown in **Figure S2**.

Taken together, these results suggest that the contrast factor captures meaningful variance in the relative predominance of knee-specific versus body-wide pain.

## Population coupling estimates

**Table 4** presents the population parameters from the Bayesian VARX(1) model. The pain-to-sleep pathway was the dominant coupling direction. The population mean (${\widehat{\lambda}}_{ps} = - 0.140$, $P({\widehat{\lambda}}_{ps} < 0) = 0.998$, 95% CrI [-0.238, -0.042]) indicated that a one-unit within-person increase in general pain predicted a 0.140-unit decrease in next-quarter sleep quality. The random effect SD was substantial (${\widehat{\tau}}_{ps} = 0.362$, 95% CrI [0.244, 0.464]), reflecting meaningful between-person heterogeneity (**Figure 2**).

**Table 4.** Population parameters from Bayesian VARX(1).

| Parameter | Description | Estimate | SD | 95% CrI | $\mathbf{P(<0)}$ | $\widehat{\mathbf{R}}$ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| ${\widehat{\mu}}_{p}$ | Pain intercept | 0.009 | 0.010 | [-0.011, 0.027] |  | 1.00 |
| ${\widehat{\varphi}}_{p}$ | **Pain autoregression** | **0.106** | **0.025** | **[0.059, 0.152]** |  | **1.00** |
| ${\widehat{\lambda}}_{sp}$ | Sleep $\rightarrow$ Pain | -0.019 | 0.018 | [-0.055, 0.017] | 0.877 | 1.00 |
| ${\widehat{\delta}}_{p}$ | Localization$\rightarrow$ Pain (direct) | 0.024 | 0.015 | [-0.004, 0.051] |  | 1.00 |
| ${\widehat{\omega}}_{sp}$ | Sleep$\times$ Localization $\rightarrow$ Pain | 0.009 | 0.018 | [-0.026, 0.043] |  | 1.00 |
| ${\widehat{\mu}}_{s}$ | Sleep intercept | 0.004 | 0.016 | [-0.025, 0.034] |  | 1.00 |
| ${\widehat{\lambda}}_{ps}$ | **Pain** $\rightarrow$ **Sleep** | **-0.141** | **0.050** | **[-0.234, -0.042]** | **0.998** | **1.00** |
| ${\widehat{\varphi}}_{s}$ | Sleep autoregression | 0.004 | 0.023 | [-0.038, 0.049] |  | 1.00 |
| ${\widehat{\delta}}_{s}$ | **Localization** $\rightarrow$ **Sleep (direct)** | **-0.050** | **0.023** | **[-0.093, -0.006]** |  | **1.00** |
| ${\widehat{\omega}}_{ps}$ | Pain$\times$ Localization $\rightarrow$ Sleep | -0.050 | 0.038 | [-0.124, 0.021] |  | 1.00 |
| ${\widehat{\tau}}_{sp}$ | SD: Sleep$\rightarrow$ Pain | 0.115 | 0.022 | [0.073, 0.156] |  | 1.01 |
| ${\widehat{\tau}}_{ps}$ | SD: Pain$\rightarrow$ Sleep | 0.362 | 0.058 | [0.244, 0.464] |  | 1.00 |
| ${\widehat{\sigma}}_{p}$ | Innovation SD (pain) | 0.437 | 0.008 | [0.424, 0.452] |  | 1.00 |
| ${\widehat{\sigma}}_{s}$ | Innovation SD (sleep) | 0.661 | 0.011 | [0.640, 0.682] |  | 1.00 |
| $\widehat{\rho}$ | **Innovation correlation** | **-0.155** | **0.024** | **[-0.202, -0.111]** |  | **1.00** |

**Note.** N = 229; 1,818 observations; 4 chains $\times$ 2,000 posterior draws (see Methods). Convergence was adequate: maximum $\widehat{R} = 1.01$; all effective sample sizes $>$ 7,000. Age and sex nuisance terms (${\widehat{\gamma}}_{age}$, ${\widehat{\gamma}}_{sex}$) are omitted from the table and none was credibly different from zero in either coupling direction. $P( < 0)$ is the one-sided posterior probability that the parameter is negative. Rows in **bold** correspond coefficients credibly different from zero.

![](../results/step04_coupling_model/step04_figure2_ps_coupling.png)

**Figure 2.** Person-specific Pain-to-Sleep coupling estimates. (A) Posterior means of person-specific coupling slopes. Each dot represents one participant; the gray diamond indicates the population mean; the dashed line marks zero. (B) Posterior mean $\pm$ 95% CrI, sorted by magnitude. Blue segments indicate negative coupling; red segments indicate positive coupling. The dashed gray line marks the population mean.

The sleep-to-pain pathway did not reach credibility at the population level (${\widehat{\lambda}}_{sp} = - 0.021$, $P({\widehat{\lambda}}_{sp} < 0) = 0.877$, 95% CrI [-0.054, 0.015]). The random effect SD (${\widehat{\tau}}_{sp} = 0.115$, 95% CrI [0.073, 0.156]) was smaller but non-negligible (**Figure 3**).

![](../results/step04_coupling_model/step04_figure3_sp_coupling.png)

**Figure 3.** Person-specific Sleep-to-Pain coupling estimates. (A) Posterior means of person-specific coupling slopes. Each dot represents one participant; the gray diamond indicates the population mean; the dashed line marks zero. (B) Posterior mean $\pm$ 95% CrI, sorted by magnitude. Blue segments indicate negative coupling; red segments indicate positive coupling. The dashed gray line marks the population mean.

Same-quarter innovations in pain and sleep quality were negatively correlated ($\widehat{\rho} = - 0.155$, $P( < 0) = 1.000$, 95% CrI [-0.202, -0.111]), indicating that within a given quarter, unexplained increases in pain co-occurred with unexplained decreases in sleep quality, or vice versa. Because this is a residual correlation after controlling for all lagged effects, it may reflect within-quarter bidirectional processes, shared contemporaneous perturbations (e.g., a flare event affecting both pain and sleep within the same assessment window), or measurement timing effects.

### Model comparison

The model comparison (see Methods: Model comparison) confirmed the dominance of the pain-to-sleep pathway. Including pain-to-sleep coupling improved predictive accuracy substantially ($\Delta{LOO}_{elpd}$ = +22.0, SE = 8.8, $\Delta$/SE = 2.49 for full vs. no-PS; $\Delta{LOO}_{elpd}$ = +23.1, SE = 9.1, $\Delta$/SE = 2.53 for no-SP vs. null), whereas including sleep-to-pain coupling yielded negligible improvement ($\Delta{LOO}_{elpd}$ = +0.7, SE = 5.5, $\Delta$/SE = 0.12 for full vs. no-SP; $\Delta{LOO}_{elpd}$ = +1.8, SE = 5.9, $\Delta$/SE = 0.30 for no-PS vs. null). Pareto $\widehat{k}$ diagnostics (60) confirmed the reliability of the LOO-CV approximation: no observation exceeded the 0.7 threshold (maximum $\widehat{k}$ = 0.68). By the conventional threshold of $|\Delta/SE| > 2$ (60), pain-to-sleep coupling substantially improved prediction while sleep-to-pain coupling did not.

### Moderation of pain location

The direct effect of pain localization on next‑quarter sleep quality was credible (${\widehat{\delta}}_{s} = - 0.050$, 95% CrI [−0.093, −0.006]; **Table 4**), indicating that quarters in which an individual’s pain was more knee‑dominant than usual were followed by poorer sleep quality. The pain × localization interaction operated in the same direction but did not reach credibility (${\widehat{\omega}}_{ps} = - 0.050$, 95% CrI [−0.124, 0.021). Together, the direct and interaction terms shifted the conditional pain‑to‑sleep coupling such that it was credibly negative at balanced and knee‑dominant localization levels, but not when pain was body‑dominant. Johnson–Neyman analysis (**Figure 4**) identified this transition at a localization value of −0.625 (−0.86 SD), with 84.6% of observations falling within the region where pain‑to‑sleep coupling was credibly present. In contrast, neither localization term moderated the sleep‑to‑pain pathway, and no Johnson–Neyman boundary was observed within the empirical range of localization values (**Figure S3**).

![](../results/step05_contrast_moderation/step05_figure4_jn_localization_ps.png)

**Figure 4.** Johnson-Neyman analysis of pain localization moderation of pain-to-sleep coupling. The blue line shows the posterior mean coupling slope as a continuous function of within-person pain localization ($K^{w}$), dashed lines show the 95% credible interval, and green shading indicates the region where the CrI excludes zero. The dotted vertical line marks the JN boundary. Vertical markers show simple slopes at body-dominant (-2 SD), balanced (0), and knee-dominant (+2 SD) localization levels with 95% CrI error bars. Blue dots show fitted coupling values (observation-level).

## Moderation of sleep-to-pain coupling by fMRI response

Although the population-average sleep-to-pain coupling was not credibly different from zero, the between-person variability was substantial relative to the fixed effect ($\left| {\widehat{\tau}}_{sp} \right|/\left| {\widehat{\lambda}}_{sp} \right| \approx$ [2.6, 5.9]), suggesting that individual-level moderators may shape the direction and magnitude of this effect and obscure a consistent population mean. We therefore tested whether BOLD activation in the ROIs proposed by Krause et al. moderated the sleep-to-pain coupling.

Left NAcc activation during painful knee stimulation was the only Krause et al. (2019) ROI that credibly moderated sleep-to-pain coupling (${\widehat{\gamma}}_{sp} = + 0.040$, 95% CrI [+0.004, +0.076]; **Table 5**): the lower the left NAcc activation during painful stimulation, the stronger the sleep-to-pain coupling. The right NAcc showed the same direction but it was not credible (${\widehat{\gamma}}_{sp} = + 0.023$, 95% CrI [-0.013, +0.057]). Johnson-Neyman analysis (**Figure 5**) revealed that sleep-to-pain coupling was credibly negative for individuals with left NAcc activation below -0.027 (near the sample mean), encompassing 49% of the sample. Above this boundary, the coupling was not credibly different from zero. JN analyses for the remaining Krause et al. (2019) ROIs is provided in the Supplementary Material (**Figure S5**).

The ACC—tested separately from the Krause et al. (2019) framework as a Sardi et al. (2024)-motivated D2-gated node—also credibly moderated sleep-to-pain coupling (${\widehat{\gamma}}_{sp} = + 0.038$, 95% CrI [+0.000, +0.077]—a marginal credibility—for the left, and (${\widehat{\gamma}}_{sp} = + 0.042$, 95% CrI [+0.003, +0.081] for the right; **Table 5**): higher pain-evoked ACC activation was associated with less negative sleep-to-pain coupling. The Johnson-Neyman analysis (**Figure 6**) revealed a pattern closely paralleling the NAcc, with credible sleep-to-pain coupling present only in individuals with below-average ACC activation. At low ACC activation ($z = - 2$), the coupling was credibly negative; at the sample mean, it was marginal; and at high activation ($z = + 2$), it was null. The low correlation between ACCs and left NAcc BOLD values ($r = 0.16$ for the left ACC and $r = 0.12$ for the right one) indicates that these two regions capture largely independent aspects of pain processing, consistent with the view that they operate as parallel rather than serial nodes within the dopaminergic system (28).

**Table 5.** fMRI stimulation BOLD moderators of sleep-to-pain coupling (N = 174).

| ROI | Framework | Expected ${\widehat{\mathbf{\gamma}}}_{\mathbf{sp}}$ | ${\widehat{\mathbf{\gamma}}}_{\mathbf{sp}}$ | 95% CrI |
|:---|:--:|:--:|:--:|:--:|
| **Left NAcc** | **Krause et al. (2019)** | **+** | **+0.040** | **[+0.004, +0.076]** |
| Right NAcc | Krause et al. (2019) | + | +0.023 | [-0.013, +0.057] |
| Contralateral S1 | Krause et al. (2019) | $-$ | -0.017 | [-0.056, +0.024] |
| Contralateral Middle Insula | Krause et al. (2019) | + | +0.017 | [-0.025, +0.060] |
| Left Thalamus | Krause et al. (2019) | + | +0.013 | [-0.025, +0.051] |
| Left Anterior Insula | Krause et al. (2019) | + | +0.006 | [-0.034, +0.050] |
| **Left dACC/MCC** | **Sardi et al. (2024)** | **+** | **+0.042** | **[+0.003, +0.081]** |
| **Right dACC/MCC** | **Sardi et al. (2024)** | **+** | **+0.038** | **[+0.000, +0.077]\*** |

**Note.** Each ROI was tested in a separate model run. The six Krause et al. (2019) ROIs test the sleep deprivation framework: sleep deprivation decreased NAcc, insula, and thalamus responses while increasing S1 reactivity. S1 and middle insula were extracted from the hemisphere contralateral to the stimulated knee (see Methods). Krause et al. (2019) defined the NAcc bilaterally ($\pm$9, 2, -7); left and right hemispheres were tested separately. The ACC ROI tests the Sardi et al. (2024) framework: ACC and NAcc as parallel D2-gated nodes. Both NAcc ROIs used GM-masked contrast images; all other ROIs used unmasked contrasts (see Methods). \*The lower CrI bound was slightly above zero, though this should be considered a marginal result given potential MCMC sampling fluctuations.

![](../results/step09_sp_jn/step09_figure5_jn_nacc.png)

**Figure 5.** Johnson-Neyman analysis of left NAcc BOLD moderation of sleep-to-pain coupling. The blue line shows the posterior mean coupling slope as a continuous function of left NAcc activation (mean contrast value within a 6 mm sphere at MNI -9, 2, -7; GM-masked), dashed lines show the 95% credible interval, and green shading indicates the region where the CrI excludes zero. The dotted vertical line marks the JN boundary. Vertical markers show simple slopes at low (Q1 - 1.5$\times$IQR), median, and high (Q3 + 1.5$\times$IQR) left NAcc levels with 95% CrI error bars. Blue dots show fitted coupling values (person-level). Rug plots show the distribution of individual left NAcc values.
![](../results/step09_sp_jn/step09_figure6_jn_acc.png)

**Figure 6.** Johnson-Neyman analyses of bilateral dACC/MCC BOLD moderation of sleep-to-pain coupling. Top panel: right dACC/MCC (MNI 6, 12, 38; 6 mm sphere; unmasked contrasts; $\hat{\gamma}_{sp}=+0.038$, 95% CrI [+0.000, +0.077]). Bottom panel: left dACC/MCC (MNI -6, 12, 38; 6 mm sphere; unmasked contrasts; $\hat{\gamma}_{sp}=+0.042$, 95% CrI [+0.003, +0.081]). In each panel, the blue line shows the posterior mean conditional coupling slope as a function of ACC activation (z-scored), dashed lines show the 95% credible interval, and green shading indicates the region where the CrI excludes zero. The dotted vertical line marks the JN boundary. Vertical markers show simple slopes at $z=-2$, $z=0$, and $z=+2$ with 95% CrI error bars. N = 174.

Finally, although the moderation was credible only for the left NAcc and both ACC among the seven sleep-to-pain ROIs (**Table 5**), all six Krause et al. (2019) univariate ${\widehat{\gamma}}_{sp}$ estimates matched the direction predicted by the sleep deprivation framework. The probability of this occurring by chance is $p = (1/2)^{6} = 0.016$ (exact sign test), providing convergent support for the Krause et al. (2019) framework at the pattern level.

## Pain-arousal relay pathway moderation

The five nodes of the Lynch et al. (2025) pain-arousal relay pathway were tested as moderators of pain-to-sleep coupling (${\widehat{\gamma}}_{ps}$) using two complementary approaches: pain-evoked fMRI response (N = 174) and grey matter volume from atlas-defined probabilistic ROIs (N = 189). Moderation was not credible for any ROI in either modality (**Table S1**)—a sensitivity analysis testing each hemisphere separately (10 models) confirmed that bilateral averaging did not mask lateralized effects. The BNST showed the strongest fMRI effect (${\widehat{\gamma}}_{ps} = - 0.077$), with the PBN—the origin of the pain-arousal relay pathway—showing the second strongest (${\widehat{\gamma}}_{ps} = - 0.075$). Notably, all five grey matter volume estimates were negative—consistent with larger arousal relay volumes predisposing stronger pain-to-sleep coupling—though no individual effect was credible (strongest: PBN, ${\widehat{\gamma}}_{ps} = - 0.062$). Johnson-Neyman analyses for each ROI are shown in **Figures S7-S8**.

# Discussion

To our knowledge, this study provides the first characterization of bidirectional within-person sleep–pain coupling at the quarterly timescale in a knee pain cohort, revealing four principal findings detailed below.

## Pain-to-sleep coupling dominates at the quarterly timescale

This finding suggests that, while day-to-day sleep fluctuations may have immediate consequences for next-day pain sensitivity—consistent sleep disruption impairing endogenous pain inhibition (20, 21) and increases pain sensitivity through peripheral and central mechanisms (22, 63)—at the quarterly timescale, accumulated pain burden may more strongly drive subsequent sleep quality through slower mediating processes such as inflammatory flare-and-remission cycles, changes in habitual sleep behavior, and seasonal shifts in activity.

A theoretical framework (64) supports the notion that quarterly coupling reflects processes distinct from—and potentially unrelated to—accumulated daily effects. Daily cross-lagged effects propagate only through autoregressive persistence; at the values typically reported (0.2–0.4) (7, 18), there is no path for daily sleep-to-pain effects to reach the next quarterly measurement. The negative innovation correlation ($\widehat{\rho}$ = −0.155) suggests that some coupling also occurs within the quarter—for example, a pain flare disrupting sleep over ensuing weeks—but because these changes unfold between consecutive assessments, they appear as co-occurring residuals rather than lagged effects, potentially underestimating coupling strength relative to daily designs.

## Knee-dominant pain worsens subsequent sleep

The pattern of simple slopes is visually consistent with a dose–response relationship, with credible coupling across 84.6% of observed contrast values and failing only when pain was strongly body-dominant; however, the within-person interaction did not reach credibility, so the gradient should be interpreted cautiously. Nevertheless, these findings suggest that interventions targeting localized pain (e.g., knee-specific) mechanisms may have downstream benefits for sleep.

## Left NAcc and ACC gate sleep-to-pain coupling

Krause et al. (2019) demonstrated that sleep deprivation amplifies pain reactivity in S1 while blunting reactivity in the NAcc, thalamus, and insula—regions involved in pain valuation, nociceptive gating, and homeostatic signaling (23). They interpreted NAcc disengagement as a failure of analgesic-relief signaling: the evaluative circuits that normally engage in descending pain modulation (24, 25) are impaired, while sensory registration is amplified, shifting the threshold for classifying stimuli as painful.

Our results mirror this pattern: individuals showing baseline left NAcc deactivation during painful knee stimulation exhibited stronger sleep-to-pain coupling over subsequent quarters, whereas this coupling was absent in those with positive BOLD responses. Importantly, Krause et al.’s (2019) findings reflect state-dependent changes under acute sleep deprivation, whereas ours suggest that trait-level differences in NAcc reward circuitry determine susceptibility to sleep-to-pain coupling over time. This trait-marker interpretation is supported by evidence that reduced NAcc reactivity during acute pain prospectively predicts the transition from subacute to chronic pain (27), identifying individuals whose reward circuitry fails to buffer nociceptive input.

We do not propose that sleep–pain coupling itself drives chronification; rather, the same NAcc phenotype that marks vulnerability to chronic pain appears to mark vulnerability to a second dynamic—the extent to which poor sleep translates into worse pain over subsequent quarters. The left-lateralized pattern is consistent with this interpretation: the left NAcc is preferentially engaged in chronic pain processing, with reduced volume in subacute and chronic low-back pain, and the transition to chronicity is specifically associated with increased left NAcc–rostral ACC connectivity and loss of left NAcc low-frequency BOLD fluctuations (26). The same lateralization extends to sleep: in hip osteoarthritis, left but not right NAcc volume is predicted by sleep quality and efficiency (65), suggesting that left NAcc structure integrates sleep-related signals. Together, these findings position the left NAcc as a trait-level node where individual differences in reward-circuit function gate how strongly sleep and pain influence each other over time, independent of chronification trajectory.

A neurochemical mechanism for this trait vulnerability is suggested by converging rodent and human evidence. In rodents, pharmacological activation of D2 receptors in the NAcc prevents sleep-restriction-induced hyperalgesia (28), identifying NAcc D2 signaling as a causal brake on sleep-induced pain amplification; complementary work shows that the hyperalgesia produced by sleep loss is mediated by reduced alertness acting through dopaminergic pathways (63). In humans, PET imaging shows that a single night of sleep deprivation downregulates D2/D3 receptors in the ventral striatum—the region containing the NAcc—and that this downregulation tracks reduced alertness (29). These findings provide a plausible neurochemical bridge between sleep loss and pain facilitation in the NAcc, and suggest that stable individual differences in dopaminergic tone—partially indexed by BOLD reactivity—could determine how strongly sleep disruption translates into pain on subsequent quarters.

The parallel moderation by pain-evoked dACC/MCC activation in both hemispheres, and especially ipsilateral to the left NAcc, is consistent with the same mechanism: Sardi et al. (28) showed that D2 activation in either NAcc or ACC independently prevents sleep-restriction hyperalgesia in rodents, framing the two as redundant D2-gated nodes.

At the pattern level, all six Krause et al. (2019) ROIs (23) showed moderation estimates in the theoretically predicted directions even though they were only credible for the left NAcc and both ACCs, providing convergent support for the framework beyond any single ROI. Causally testing this circuit—for example, by combining D2 pharmacological manipulation with longitudinal sleep–pain monitoring—would clarify whether the trait vulnerability we observe is dopaminergically modifiable.

## Arousal relay nodes show directional but non-credible pain-to-sleep moderation

The pain-arousal relay analysis did not yield credible moderation of pain-to-sleep coupling by any individual arousal relay node, whether indexed by pain-evoked BOLD response or grey matter volume. However, grey matter volumes of all five nodes showed moderation estimates in the same direction, suggesting a consistent pattern that a composite measure (e.g., PCA across the five nodes) might detect—an approach beyond the theory-driven, individual-ROI framework adopted here. The interpretation of the direction is also ambiguous—larger volume could reflect greater arousal capacity or preserved structure in the absence of atrophy.

## Methodological considerations

The present model addresses several methodological limitations of the existing daily sleep-pain literature (see **Supplementary Note S1** for more detailed methodological considerations). Most daily studies estimate each coupling direction in separate regression equations, whether multilevel regressions (3–7, 10, 11, 14–17, 66), multilevel structural equation models (12, 13), or bivariate correlations (67), leaving the cross-lagged coefficients unconditioned on each other and the contemporaneous covariance between innovations unmodeled. Only Edwards et al. (2008)—and the one using month-long lags (9)—estimated both directions simultaneously; Edwards et al.'s SEM also likely included correlated residuals by default (18), but the value was not reported. Furthermore, neither decomposed variance into within- and between-person components—a step taken by roughly half the daily literature through person-mean centering or equivalent methods (3, 4, 6, 7, 10–16) but omitted by others (2, 9, 17, 18), risking inflation of cross-lagged estimates from trait-level confounds (19). Note that the within-person is analogous to the random intercept in the RI-CLPM (58, 68), thus addressing the bias in standard CLPMs (19).

The present model estimates both directions simultaneously under Bayesian partial pooling (which regularizes noisy individual-level estimates toward the population mean), explicitly models the contemporaneous covariance between pain and sleep innovations via a Cholesky parameterization, and adopts the within-person centering. Finally, it also includes person-specific random slopes on the cross-lagged coefficients, capturing individual heterogeneity in coupling strength that a single population-average path cannot represent, a feature not included in any prior daily study. Although a few studies have tested between-person moderators of the within-person sleep-pain association—baseline depression (14), group membership (5), and daily mood (6, 16)—these tested whether a moderator shifts the population-average slope, not whether it explains individual variation in coupling strength. The random slopes are what make the latter possible, and the substantial heterogeneity observed motivates the search for moderators.

## Limitations

With 229 participants and 1,818 lagged transitions, the present study is substantially larger than most daily diary studies (typically N = 30–100), apart from Edwards et al. (2008; N = 971), which drew from the general population rather than a clinical cohort (18). It is, however, smaller than prospective epidemiological cohorts examining sleep–pain associations over years of follow-up (69). The fMRI subsample (N = 174; 76% of the full sample) further reduces statistical power for moderation analyses.

The within-person centering also removes overall severity from the model: a person fluctuating around 8/10 pain contributes identically to the coupling estimate as one fluctuating around 3/10, because both are modeled in deviations from their respective baselines. This raises the question of whether coupling strength depends on a person's overall severity level. To test whether coupling strength depends on overall severity, we entered person-mean pain severity and person-mean sleep quality as between-person moderators of both coupling directions, individually and simultaneously (**Table S2**). None of the six moderation parameters were credibly different from zero, indicating that the within-person coupling process operates comparably regardless of baseline severity.

Sleep quality was assessed with a single subjective item, which does not distinguish dimensions such as efficiency, duration, or architecture, a limitation given REM sleep characteristics are associated with pain sensitivity (70) and sleep continuity disruption impairs endogenous pain inhibition (20). Multi-item instruments, actigraphy, or polysomnography would better characterize the sleep dimensions most relevant to pain coupling. That said, in musculoskeletal pain populations, the sleep-to-pain association is stronger for self-reported sleep than for actigraphy-derived measures (8).

Quarterly measurement may be suboptimal for the coupling process. Optimal lag analysis suggests the quarterly interval captures roughly 32% of the peak unidirectional cross-lagged signal, and weekly or biweekly assessments would substantially increase sensitivity (**Supplementary Note S1**). The coupling patterns observed here reflect cumulative quarterly processes and should not be directly compared with daily diary findings.

Although participants also underwent MRI at the two-year follow-up, we did not use these data because 37% attrition would have further reduced the already limited neuroimaging subsample. Also, the left NAcc and ACC moderation findings were not corrected for multiple comparisons; however, all ROIs were selected a priori from published frameworks with pre-specified directional predictions, and the convergent pattern across independently defined ROIs from distinct theoretical frameworks strengthens the evidence for reward-circuit gating of sleep-to-pain coupling. Additionally, the pain-arousal relay pathway (31) was identified in mice using optogenetics, and the human homologues of these small subcortical structures are difficult to resolve at standard 3 mm fMRI resolution, with some ROIs containing as few as 3 voxels (e.g., PBN, LH). Higher-resolution fMRI, dedicated brainstem protocols, and larger samples would be needed to adequately test this pathway in humans.

# Data Availability

The data that support the findings of this study contain protected health information from human subjects and cannot be shared publicly due to IRB and HIPAA restrictions. De-identified data are available from the corresponding author upon reasonable request, subject to institutional data use agreements. Analysis code is available at <https://github.com/pvaldeshernandez/quarterly_sleep-pain_coupling.>

# Acknowledgements

We are grateful to our participants and study teams at the University of Florida (UF) and the University of Alabama, Birmingham (UAB).

**Use of artificial intelligence.** Portions of this manuscript were drafted, edited, and checked with the assistance of Claude Opus 4.6 (Anthropic, San Francisco, CA). The AI was used to polish prose, code development and refactoring, citation verification, and generation of figures under the direct supervision of the first author, who tested all codes independently and takes full responsibility for the accuracy of all content.

# Funding

This work was supported by NIH/NIA Grants K01AG083228 (PAVH); R01AG059809, R01AG067757 (YCA); R37AG033906 (RBF). A portion of this work was performed in the McKnight Brain Institute at the National High Magnetic Field Laboratory's Advanced Magnetic Resonance Imaging and Spectroscopy (AMRIS) Facility, which is supported by National Science Foundation Cooperative Agreement No. DMR-1157490 and DMR-1644779 and the State of Florida. This content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health or other funding agencies.

# CRediT Author Statement

# References

1. M. T. Smith, J. A. Haythornthwaite, How do sleep disturbance and chronic pain inter-relate? Insights from the longitudinal and cognitive-behavioral clinical trials literature. *Sleep Med. Rev.* **8**, 119–132 (2004).

2. P. H. Finan, B. R. Goodin, M. T. Smith, The Association of Sleep and Pain: An Update and a Path Forward. *J. Pain* **14**, 1539–1552 (2013).

3. J. I. Gerhart, *et al.*, Relationships Between Sleep Quality and Pain-Related Factors for People with Chronic Low Back Pain: Tests of Reciprocal and Time of Day Effects. *Annals of Behavioral Medicine* **51**, 365–375 (2017).

4. D. Whibley, T. J. Braley, A. L. Kratz, S. L. Murphy, Transient Effects of Sleep on Next-Day Pain and Fatigue in Older Adults With Symptomatic Osteoarthritis. *J. Pain* **20**, 1373–1382 (2019).

5. A. S. Lewandowski, T. M. Palermo, S. De la Motte, R. Fu, Temporal daily associations between pain and sleep in adolescents with chronic pain versus healthy adolescents. *Pain* **151**, 220–225 (2010).

6. M. H. Bromberg, K. M. Gil, L. E. Schanberg, Daily sleep quality and mood as predictors of pain in children with juvenile polyarticular arthritis. *Health Psychology* **31**, 202–209 (2012).

7. K. Abeler, S. Bergvik, T. Sand, O. Friborg, Daily associations between sleep and pain in patients with chronic musculoskeletal pain. *J. Sleep Res.* **30** (2021).

8. Z. Goossens, *et al.*, Day-to-day associations between pain intensity and sleep outcomes in an adult chronic musculoskeletal pain population: A systematic review. *Sleep Med. Rev.* **79**, 102013 (2025).

9. P. J. Quartana, E. M. Wickwire, B. Klick, E. Grace, M. T. Smith, Naturalistic changes in insomnia symptoms and pain in temporomandibular joint disorder: A cross-lagged panel analysis. *Pain* **149**, 325–331 (2010).

10. G. Affleck, S. Urrows, H. Tennen, P. Higgins, M. Abeles, Sequential daily relations of sleep, pain intensity, and attention to pain among women with fibromyalgia. *Pain* **68**, 363–368 (1996).

11. J. M. Dzierzewski, *et al.*, Daily Variations in Objective Nighttime Sleep and Subjective Morning Pain in Older Adults with Insomnia: Evidence of Covariation over Time. *J. Am. Geriatr. Soc.* **58**, 925–930 (2010).

12. D. J. Kothari, M. C. Davis, E. W. Yeung, H. A. Tennen, Positive affect and pain: mediators of the within-day relation linking sleep quality to activity interference in fibromyalgia. *Pain* **156**, 540–546 (2015).

13. C. J. Mun, *et al.*, Pain Expectancy and Positive Affect Mediate the day-to-day Association Between Objectively Measured Sleep and Pain Severity Among Women With Temporomandibular Disorder. *J. Pain* **23**, 669–679 (2022).

14. E. M. O’Brien, *et al.*, Intraindividual Variability in Daily Sleep and Pain Ratings Among Chronic Pain Patients. *Clin. J. Pain* **27**, 425–433 (2011).

15. N. K. Y. Tang, C. E. Goodchild, A. N. Sanborn, J. Howard, P. M. Salkovskis, Deciphering the Temporal Link between Pain and Sleep in a Heterogeneous Chronic Pain Patient Sample: A Multilevel Daily Process Study. *Sleep* **35**, 675–687 (2012).

16. C. R. Valrie, K. M. Gil, R. Redding-Lallinger, C. Daeschner, Brief Report: Daily Mood as a Mediator or Moderator of the Pain-Sleep Relationship in Children with Sickle Cell Disease. *J. Pediatr. Psychol.* **33**, 317–322 (2007).

17. S. M. Alsaadi, *et al.*, The Bidirectional Relationship Between Pain Intensity and Sleep Disturbance/Quality in Patients With Low Back Pain. *Clin. J. Pain* **30**, 755–765 (2014).

18. R. R. Edwards, D. M. Almeida, B. Klick, J. A. Haythornthwaite, M. T. Smith, Duration of sleep contributes to next-day pain report in the general population ☆. *Pain* **137**, 202–207 (2008).

19. R. E. Lucas, Why the Cross-Lagged Panel Model Is Almost Never the Right Choice. *Adv. Methods Pract. Psychol. Sci.* **6** (2023).

20. M. T. Smith, R. R. Edwards, U. D. McCann, J. A. Haythornthwaite, The Effects of Sleep Deprivation on Pain Inhibition and Spontaneous Pain in Women. *Sleep* **30**, 494–505 (2007).

21. T. Roehrs, M. Hyde, B. Blaisdell, M. Greenwald, T. Roth, Sleep Loss and REM Sleep Loss are Hyperalgesic. *Sleep* **29**, 145–151 (2006).

22. K. Kourbanova, C. Alexandre, A. Latremoliere, Effect of sleep loss on pain—New conceptual and mechanistic avenues. *Front. Neurosci.* **16** (2022).

23. A. J. Krause, A. A. Prather, T. D. Wager, M. A. Lindquist, M. P. Walker, The Pain of Sleep Loss: A Brain Characterization in Humans. *The Journal of Neuroscience* **39**, 2291–2300 (2019).

24. M. N. Baliki, P. Y. Geha, H. L. Fields, A. V. Apkarian, Predicting Value of Pain and Analgesia: Nucleus Accumbens Response to Noxious Stimuli Changes in the Presence of Chronic Pain. *Neuron* **66**, 149–160 (2010).

25. E. Navratilova, F. Porreca, Reward and motivation in pain and pain relief. *Nat. Neurosci.* **17**, 1304–1312 (2014).

26. M. M. Makary, *et al.*, Loss of nucleus accumbens low-frequency fluctuations is a signature of chronic pain. *Proceedings of the National Academy of Sciences* **117**, 10015–10023 (2020).

27. M. N. Baliki, *et al.*, Corticostriatal functional connectivity predicts transition to chronic back pain. *Nat. Neurosci.* **15**, 1117–1119 (2012).

28. N. F. Sardi, *et al.*, Sleep and Pain: A Role for the Anterior Cingulate Cortex, Nucleus Accumbens, and Dopamine in the Increased Pain Sensitivity Following Sleep Restriction. *J. Pain* **25**, 331–349 (2024).

29. N. D. Volkow, *et al.*, Evidence That Sleep Deprivation Downregulates Dopamine D2R in Ventral Striatum in the Human Brain. *The Journal of Neuroscience* **32**, 6711–6717 (2012).

30. H. Ito, *et al.*, Chronic pain recruits hypothalamic dynorphin/kappa opioid receptor signalling to promote wakefulness and vigilance. *Brain* **146**, 1186–1199 (2023).

31. N. Lynch, *et al.*, Calcitonin Gene‐Related Peptide (CGRP)‐Expressing Neurons in the External Lateral Parabrachial Area Regulate Pain‐Induced Sleep Disturbances. *Advanced Science* **12** (2025).

32. M. Davis, D. L. Walker, L. Miles, C. Grillon, Phasic vs Sustained Fear in Rats and Humans: Role of the Extended Amygdala in Fear vs Anxiety. *Neuropsychopharmacology* **35**, 105–135 (2010).

33. P. A. Valdes-Hernandez, *et al.*, Widespread and prolonged pain may reduce brain clearance capacity only via sleep impairment: Evidence from participants with knee pain. *J. Pain* **30**, 105356 (2025).

34. P. A. Valdes-Hernandez, *et al.*, Accelerated Brain Aging Mediates the Association Between Psychological Profiles and Clinical Pain in Knee Osteoarthritis. *J. Pain* **25**, 104423 (2024).

35. A. J. Johnson, *et al.*, Psychological profiles in adults with knee OA-related pain: a replication study. *Ther. Adv. Musculoskelet. Dis.* **13** (2021).

36. P. A. Valdes-Hernandez, *et al.*, Brain-predicted age difference estimated using DeepBrainNet is significantly associated with pain and function—a multi-institutional and multiscanner study. *Pain* **164**, 2822–2838 (2023).

37. E. L. Terry, *et al.*, Associations of pain catastrophizing with pain-related brain structure in individuals with or at risk for knee osteoarthritis: Sociodemographic considerations. *Brain Imaging Behav.* **15**, 1769–1777 (2021).

38. S. Booker, *et al.*, Movement-evoked pain, physical function, and perceived stress: An observational study of ethnic/racial differences in aging non-Hispanic Blacks and non-Hispanic Whites with knee osteoarthritis. *Exp. Gerontol.* **124**, 110622 (2019).

39. E. J. Bartley, *et al.*, Race/Ethnicity Moderates the Association Between Psychosocial Resilience and Movement‐Evoked Pain in Knee Osteoarthritis. *ACR Open Rheumatol.* **1**, 16–25 (2019).

40. J. J. Tanner, *et al.*, More than chronic pain: behavioural and psychosocial protective factors predict lower brain age in adults with/at risk of knee osteoarthritis over two years. *Brain Commun.* **7** (2025).

41. F. A. Huber, *et al.*, Neighborhood Disadvantage and Knee Osteoarthritis Pain: Do Sleep and Catastrophizing Play a Role? *Arthritis Care Res. (Hoboken).* **77**, 95–103 (2025).

42. A. J. Johnson, J. Cole, R. B. Fillingim, Y. Cruz-Almeida, Persistent Non-pharmacological Pain Management and Brain-Predicted Age Differences in Middle-Aged and Older Adults With Chronic Knee Pain. *Frontiers in Pain Research* **3** (2022).

43. A. J. Johnson, *et al.*, Cross-Sectional Brain-Predicted Age Differences in Community-Dwelling Middle-Aged and Older Adults with High Impact Knee Pain. *J. Pain Res.* **Volume 15**, 3575–3587 (2022).

44. J. L. Horn, A Rationale and Test for the Number of Factors in Factor Analysis. *Psychometrika* **30**, 179–185 (1965).

45. N. Bellamy, W. W. Buchanan, C. H. Goldsmith, J. Campbell, L. W. Stitt, Validation study of WOMAC: a health status instrument for measuring clinically important patient relevant outcomes to antirheumatic drug therapy in patients with osteoarthritis of the hip or knee. *J. Rheumatol.* **15**, 1833–40 (1988).

46. J. H. Kellgren, J. S. Lawrence, Radiological Assessment of Osteo-Arthrosis. *Ann. Rheum. Dis.* **16**, 494–502 (1957).

47. J. Ashburner, K. J. Friston, Unified segmentation. *Neuroimage* **26**, 839–851 (2005).

48. J. Ashburner, A fast diffeomorphic image registration algorithm. *Neuroimage* **38**, 95–113 (2007).

49. P. D. Youell, *et al.*, Lateralisation of nociceptive processing in the human brain: a functional magnetic resonance imaging study. *Neuroimage* **23**, 1068–1077 (2004).

50. A. Xu, *et al.*, Convergent neural representations of experimentally-induced acute pain in healthy volunteers: A large-scale fMRI meta-analysis. *Neurosci. Biobehav. Rev.* **112**, 300–323 (2020).

51. L. L. Symonds, N. S. Gordon, J. C. Bixby, M. M. Mande, Right-Lateralized Pain Processing in the Human Cortex: An fMRI Study. *J. Neurophysiol.* **95**, 3823–3830 (2006).

52. M. Laubach, L. M. Amarante, K. Swanson, S. R. White, What, If Anything, Is Rodent Prefrontal Cortex? *eNeuro* **5**, ENEURO.0315-18.2018 (2018).

53. K. Singh, *et al.*, Structural connectivity of autonomic, pain, limbic, and sensory brainstem nuclei in living humans based on 7 Tesla and 3 Tesla MRI. *Hum. Brain Mapp.* **43**, 3086–3112 (2022).

54. L. Zaborszky, *et al.*, Stereotaxic probabilistic maps of the magnocellular cell groups in human basal forebrain. *Neuroimage* **42**, 1127–1141 (2008).

55. W. M. Pauli, A. N. Nili, J. M. Tyszka, A high-resolution probabilistic in vivo atlas of human subcortical brain nuclei. *Sci. Data* **5**, 180063 (2018).

56. J. D. Theiss, C. Ridgewell, M. McHugo, S. Heckers, J. U. Blackford, Manual segmentation of the human bed nucleus of the stria terminalis using 3 T MRI. *Neuroimage* **146**, 288–292 (2017).

57. C. Neudorfer, *et al.*, A high-resolution in vivo magnetic resonance imaging atlas of the human hypothalamic region. *Sci. Data* **7**, 305 (2020).

58. P. J. Curran, D. J. Bauer, The Disaggregation of Within-Person and Between-Person Effects in Longitudinal Models of Change. *Annu. Rev. Psychol.* **62**, 583–619 (2011).

59. O. Abril-Pla, *et al.*, PyMC: a modern, and comprehensive probabilistic programming framework in Python. *PeerJ Comput. Sci.* **9**, e1516 (2023).

60. A. Vehtari, A. Gelman, J. Gabry, Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Stat. Comput.* **27**, 1413–1432 (2017).

61. R. Kumar, C. Carroll, A. Hartikainen, O. Martin, ArviZ a unified library for exploratory analysis of Bayesian models in Python. *J. Open Source Softw.* **4**, 1143 (2019).

62. A. F. Hayes, *Introduction to Mediation, Moderation, and Conditional Process Analysis*, Second Edi (The Guilford Press, 2018).

63. C. Alexandre, *et al.*, Decreased alertness due to sleep loss increases pain sensitivity in mice. *Nat. Med.* **23**, 768–774 (2017).

64. C. Dormann, M. A. Griffin, Optimal time lags in panel studies. *Psychol. Methods* **20**, 489–505 (2015).

65. N. Egorova-Brumley, *et al.*, Left nucleus accumbens volume is associated with poor sleep in hip osteoarthritis. *Neurobiology of Pain* **18**, 100203 (2025).

66. C. S. Mccrae, *et al.*, Sleep and affect in older adults: Using multilevel modeling to examine daily associations. *J. Sleep Res.* **17**, 42–53 (2008).

67. J. J. Liszka-Hackzell, D. P. Martin, Analysis of Nighttime Activity and Daytime Pain in Patients with Chronic Back Pain Using a Self-Organizing Map Neural Network. *J. Clin. Monit. Comput.* **19**, 411–414 (2005).

68. E. L. Hamaker, R. M. Kuiper, R. P. P. P. Grasman, A critique of the cross-lagged panel model. *Psychol. Methods* **20**, 102–116 (2015).

69. E. F. Afolalu, F. Ramlee, N. K. Y. Tang, Effects of sleep changes on pain-related health outcomes in the general population: A systematic review of longitudinal studies with exploratory meta-analysis. *Sleep Med. Rev.* **39**, 82–97 (2018).

70. M. T. Smith, R. R. Edwards, G. L. Stonerock, U. D. McCann, Individual Variation in Rapid Eye Movement Sleep Is Associated With Pain Perception in Healthy Women: Preliminary Data. *Sleep* **28**, 809–812 (2005).

