function job = create_first_level_job(func_dir, output_dir, motion_file, wm_signal, csf_signal, mask_file)
% CREATE_FIRST_LEVEL_JOB  Build an SPM12 batch for a single-subject first-level GLM.
%
%   job = create_first_level_job(func_dir, output_dir, motion_file, ...
%                                wm_signal, csf_signal, mask_file)
%
%   This function constructs a complete SPM12 batch structure that specifies
%   and estimates a first-level (single-subject) general linear model for a
%   von Frey pain stimulation fMRI experiment. The batch is returned as a
%   cell array suitable for spm_jobman('run', job).
%
%   NO CONN DEPENDENCY.  The original analysis used the CONN toolbox to set
%   up first-level GLMs.  This function replicates that specification using
%   only SPM12 batch calls, making the pipeline fully reproducible without
%   CONN.
%
% =========================================================================
%   PARADIGM: Von Frey pain stimulation
% =========================================================================
%   Participants with knee osteoarthritis (OA) received quantitative sensory
%   testing (QST) during fMRI acquisition.  A calibrated von Frey filament
%   was applied to the most painful knee in a block design:
%
%       Block 1:   24 -  48 s   (stimulation)
%       Block 2:   96 - 120 s   (stimulation)
%       Block 3:  168 - 192 s   (stimulation)
%       Block 4:  240 - 264 s   (stimulation)
%       Block 5:  312 - 336 s   (stimulation)
%
%   Each block lasts 24 seconds.  The inter-block intervals (~48 s) serve as
%   the implicit baseline (rest).  Total acquisition: ~6 minutes, 150
%   volumes at TR = 2.4 s.
%
%   Onsets  = [24, 96, 168, 240, 312] seconds
%   Duration = 24 seconds per block
%
% =========================================================================
%   MODEL SPECIFICATION
% =========================================================================
%   The GLM contains a single task regressor (the five stimulation blocks
%   convolved with SPM's canonical hemodynamic response function) plus eight
%   nuisance regressors organized in a matrix R:
%
%     Columns 1-6:  Six rigid-body motion parameters from realignment
%                   (three translations in mm: x, y, z; three rotations in
%                   radians: pitch, roll, yaw).  These capture variance due
%                   to residual head motion after realignment, which can
%                   introduce spurious signal changes at tissue boundaries.
%
%     Column 7:     Average white matter (WM) BOLD signal.  WM tissue should
%                   not contain task-related neural signal; its timecourse
%                   captures physiological noise (cardiac and respiratory
%                   artifacts) and scanner-related drift.  Regressing out WM
%                   signal improves sensitivity to true neural activation.
%
%     Column 8:     Average cerebrospinal fluid (CSF) BOLD signal.  Like WM,
%                   CSF voxels contain no neural signal but are sensitive to
%                   physiological pulsations (especially cardiac).  Including
%                   CSF as a nuisance regressor removes a major source of
%                   non-neural variance from the residuals.
%
%   HIGH-PASS FILTER (128 s):
%     A discrete cosine transform (DCT) basis set with a period cutoff of
%     128 seconds removes low-frequency drift from both the data and the
%     design matrix.  This is the SPM default and is appropriate for block
%     designs: the shortest inter-onset interval here is 72 s, so the task
%     fundamental period (~72 s) is well above the 128 s cutoff and is fully
%     preserved.  Drift from scanner heating, physiological slow oscillations,
%     and subject movement trends is attenuated.
%
%   SERIAL CORRELATIONS - AR(1):
%     BOLD fMRI time series exhibit positive temporal autocorrelation: each
%     volume is correlated with the next due to hemodynamic sluggishness and
%     physiological rhythms.  If ignored, the residual variance is
%     underestimated and t-statistics are inflated.  SPM's AR(1) model
%     estimates a single autoregressive parameter at each voxel and
%     "prewhitens" both the data and design matrix, yielding valid standard
%     errors.  (Reference: Friston KJ, et al. Statistical Parametric
%     Mapping: The Analysis of Functional Brain Images. Academic Press, 2007.)
%
%   HEMODYNAMIC RESPONSE FUNCTION:
%     We use SPM's canonical HRF (a double-gamma function peaking at ~5 s)
%     without time or dispersion derivatives.  For a well-characterized block
%     design with long stimulation epochs (24 s), the canonical HRF alone
%     captures the vast majority of the evoked signal.
%
%   MICROTIME PARAMETERS:
%     Microtime resolution = 16 bins per TR.  This is the temporal resolution
%     at which the regressors are constructed before downsampling to the
%     acquired TR.  Microtime onset = 1, meaning the regressors are sampled
%     at the time of the first slice.
%
% =========================================================================
%   INPUTS
% =========================================================================
%   func_dir    - [string] Path to directory containing the preprocessed
%                 4D functional NIfTI (e.g., 'sw*.nii').  The function
%                 will find all NIfTI files matching 'sw*.nii' in this
%                 directory and expand them to individual volume-level
%                 entries (SPM requires volume-by-volume file lists).
%
%   output_dir  - [string] Directory where SPM.mat and beta images will be
%                 written.  Created if it does not exist.
%
%   motion_file - [string] Path to the rp_*.txt realignment parameter file.
%                 This is a 150 x 6 text file produced by SPM's realignment
%                 step, with columns: [x y z pitch roll yaw].
%
%   wm_signal   - [150 x 1 double] Average white matter BOLD signal across
%                 time, extracted from the subject's WM mask.
%
%   csf_signal  - [150 x 1 double] Average CSF BOLD signal across time,
%                 extracted from the subject's CSF mask.
%
%   mask_file   - [string] Path to an explicit mask image (e.g., a
%                 binarized grey matter mask in MNI space), or '' (empty
%                 string) for no explicit mask.  When a GM mask is used,
%                 SPM restricts estimation to voxels inside the mask.
%                 When no mask is used (''), SPM applies its default
%                 implicit mask (threshold = 0.8 * global mean), which
%                 yields whole-brain coverage including subcortical
%                 structures that may be excluded by a cortical GM mask.
%
% =========================================================================
%   OUTPUT
% =========================================================================
%   job  - [cell array] SPM12 batch structure with two modules:
%          {1} spm.stats.fmri_spec  - model specification
%          {2} spm.stats.fmri_est   - model estimation
%          Ready to run via: spm_jobman('run', job)
%
% =========================================================================
%   USAGE EXAMPLE
% =========================================================================
%   % Specify paths for one subject
%   func_dir    = '/data/study/sub-001/func';
%   output_dir  = '/data/study/derivatives/first_level/sub-001';
%   motion_file = '/data/study/sub-001/func/rp_sub-001.txt';
%   wm_signal   = load('/data/study/sub-001/wm_timecourse.txt');
%   csf_signal  = load('/data/study/sub-001/csf_timecourse.txt');
%   mask_file   = '/data/study/derivatives/gm_mask.nii';
%
%   job = create_first_level_job(func_dir, output_dir, motion_file, ...
%                                wm_signal, csf_signal, mask_file);
%   spm_jobman('run', job);
%
% =========================================================================
%   REFERENCE
% =========================================================================
%   Friston KJ, Ashburner JT, Kiebel SJ, Nichols TE, Penny WD (eds).
%   Statistical Parametric Mapping: The Analysis of Functional Brain Images.
%   Academic Press, 2007.
%
% -------------------------------------------------------------------------
%   Part of: quarterly_sleep-pain_coupling
%   See also: run_first_level_glm, create_stim_vs_baseline_contrast
% -------------------------------------------------------------------------

% =====================================================================
%   INPUT VALIDATION
% =====================================================================

% Check that the functional directory exists
if ~exist(func_dir, 'dir')
    error('create_first_level_job:BadInput', ...
          'Functional directory does not exist: %s', func_dir);
end

% Create output directory if needed
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Check that the motion parameter file exists
if ~exist(motion_file, 'file')
    error('create_first_level_job:BadInput', ...
          'Motion parameter file not found: %s', motion_file);
end

% Validate nuisance regressor dimensions
if ~isnumeric(wm_signal) || ~isnumeric(csf_signal)
    error('create_first_level_job:BadInput', ...
          'wm_signal and csf_signal must be numeric vectors.');
end
wm_signal = wm_signal(:);   % ensure column vector
csf_signal = csf_signal(:); % ensure column vector

% =====================================================================
%   FIND FUNCTIONAL VOLUMES
% =====================================================================
% SPM batch expects a cell array of file paths, one per volume.
% For a 4D NIfTI, we expand it to 'file.nii,1', 'file.nii,2', etc.
% The preprocessed files are named with the 'sw' prefix (smoothed,
% warped to MNI space via DARTEL).

func_files = spm_select('ExtFPList', func_dir, '^sw.*\.nii$', Inf);
%   spm_select('ExtFPList', ...) returns a character array where each row
%   is one volume-level filename (e.g., 'path/swfile.nii,1').  The third
%   argument is a regular expression filter; the fourth (Inf) tells SPM
%   to expand all frames from all matching 4D files.

if isempty(func_files)
    error('create_first_level_job:NoFiles', ...
          'No smoothed functional files (sw*.nii) found in: %s', func_dir);
end

% Convert the character array to a cell array of strings (SPM batch format)
func_files = cellstr(func_files);

fprintf('  Found %d functional volumes in %s\n', numel(func_files), func_dir);

% =====================================================================
%   BUILD THE NUISANCE REGRESSOR MATRIX (R)
% =====================================================================
% SPM's "multiple regressors" field expects a .mat file containing a
% variable named R.  R is an [n_volumes x n_regressors] matrix.
%
% We construct R with 8 columns:
%   Columns 1-6: motion parameters (from realignment)
%   Column 7:    average white matter BOLD signal
%   Column 8:    average CSF BOLD signal

motion_params = load(motion_file);
%   The rp_*.txt file is a plain-text file with 150 rows and 6 columns.
%   Columns: [x_translation, y_translation, z_translation, ...
%             pitch_rotation, roll_rotation, yaw_rotation]
%   Units: mm for translations, radians for rotations.

n_volumes = size(motion_params, 1);

% Sanity check: WM and CSF timecourses must match the number of volumes
if numel(wm_signal) ~= n_volumes
    error('create_first_level_job:BadInput', ...
          'wm_signal has %d elements but expected %d (n_volumes).', ...
          numel(wm_signal), n_volumes);
end
if numel(csf_signal) ~= n_volumes
    error('create_first_level_job:BadInput', ...
          'csf_signal has %d elements but expected %d (n_volumes).', ...
          numel(csf_signal), n_volumes);
end

% Assemble the 8-column nuisance matrix
R = [motion_params, wm_signal, csf_signal]; %#ok<NASGU>
%   R is now [150 x 8].  The variable name must be exactly 'R' because
%   SPM's batch system looks for a variable with this name when loading
%   the multiple regressors .mat file.

% Save the nuisance regressors to a .mat file in the output directory
regressors_file = fullfile(output_dir, 'nuisance_regressors.mat');
save(regressors_file, 'R');
%   This file will be referenced by the batch specification below.

% =====================================================================
%   MODULE 1: fMRI MODEL SPECIFICATION (spm.stats.fmri_spec)
% =====================================================================
% This module tells SPM how to build the design matrix (X) for the GLM.
% The design matrix has one row per volume and one column per regressor
% (task + nuisance + constant).

% --- Timing parameters ---
job{1}.spm.stats.fmri_spec.dir            = {output_dir};
%   Where to write SPM.mat and beta images.

job{1}.spm.stats.fmri_spec.timing.units   = 'secs';
%   Onsets and durations are specified in seconds (not scans).

job{1}.spm.stats.fmri_spec.timing.RT      = 2.4;
%   Repetition time (TR) = 2.4 seconds.  This is the time between
%   successive volume acquisitions.

job{1}.spm.stats.fmri_spec.timing.fmri_t  = 16;
%   Microtime resolution: the number of time bins per TR used to
%   construct the convolved regressors.  A value of 16 provides
%   sufficient temporal resolution (2.4 / 16 = 0.15 s per bin) to
%   accurately model the hemodynamic response.

job{1}.spm.stats.fmri_spec.timing.fmri_t0 = 1;
%   Microtime onset: the bin within each TR at which the regressors
%   are sampled.  A value of 1 means the regressors are evaluated at
%   the time of the first slice (consistent with slice timing
%   correction to the first slice).

% --- Session data ---
% A "session" in SPM corresponds to one contiguous run of fMRI data.
% Our paradigm has a single run per subject.

job{1}.spm.stats.fmri_spec.sess(1).scans  = func_files;
%   Cell array of volume-level filenames (150 entries for 150 volumes).

% --- Task condition: Stimulation ---
job{1}.spm.stats.fmri_spec.sess(1).cond(1).name     = 'Stimulation';
%   Human-readable label for this condition.

job{1}.spm.stats.fmri_spec.sess(1).cond(1).onset    = [24 96 168 240 312];
%   Onset times (in seconds) of each stimulation block.
%   Block 1 starts at 24 s, Block 2 at 96 s, Block 3 at 168 s,
%   Block 4 at 240 s, Block 5 at 312 s.

job{1}.spm.stats.fmri_spec.sess(1).cond(1).duration = [24 24 24 24 24];
%   Duration of each block in seconds.  All blocks are 24 s long.

job{1}.spm.stats.fmri_spec.sess(1).cond(1).tmod     = 0;
%   Time modulation: 0 = none.  We do not model changes in the
%   response amplitude across blocks (no parametric modulation by time).

job{1}.spm.stats.fmri_spec.sess(1).cond(1).pmod     = struct('name', {}, ...
                                                              'param', {}, ...
                                                              'poly', {});
%   Parametric modulators: none.  This empty struct tells SPM that
%   there are no parametric modulators for this condition.

job{1}.spm.stats.fmri_spec.sess(1).cond(1).orth     = 1;
%   Orthogonalize parametric modulators: 1 = yes (default).
%   Irrelevant here since we have no parametric modulators, but
%   included for completeness.

% --- Nuisance regressors ---
% We provide the 8-column nuisance matrix via a .mat file.
job{1}.spm.stats.fmri_spec.sess(1).multi          = {''};
%   Multiple conditions file: not used (we specify conditions inline).

job{1}.spm.stats.fmri_spec.sess(1).regress        = struct('name', {}, ...
                                                            'val', {});
%   Individual regressors specified inline: none (we use the .mat file).

job{1}.spm.stats.fmri_spec.sess(1).multi_reg      = {regressors_file};
%   Path to the .mat file containing the R matrix (8 nuisance regressors).

job{1}.spm.stats.fmri_spec.sess(1).hpf            = 128;
%   High-pass filter cutoff in seconds.  A 128 s cutoff removes
%   frequencies below 1/128 Hz = 0.0078 Hz.  This attenuates:
%     - Scanner drift (thermal changes in the MR hardware)
%     - Very slow physiological fluctuations (Mayer waves, ~0.1 Hz
%       vasomotor oscillations are above this cutoff and preserved)
%     - Subject movement trends
%   The task fundamental frequency (~1/72 Hz = 0.014 Hz, from the
%   shortest inter-onset interval) is well above the cutoff and is
%   unaffected.

% --- Basis function: Canonical HRF ---
job{1}.spm.stats.fmri_spec.fact                    = struct('name', {}, ...
                                                            'levels', {});
%   Factorial design: none.  We have a simple single-condition design.

job{1}.spm.stats.fmri_spec.bases.hrf.derivs        = [0 0];
%   HRF derivatives: [time_derivative, dispersion_derivative] = [0, 0].
%   We use ONLY the canonical HRF without derivatives because:
%     (a) Block designs are robust to HRF latency variations (the long
%         24 s blocks ensure the plateau region dominates the response).
%     (b) Adding derivatives increases model complexity and can absorb
%         true signal into nuisance components.
%     (c) The original CONN pipeline used the canonical HRF alone.

% --- Volterra interactions ---
job{1}.spm.stats.fmri_spec.volt                    = 1;
%   Model interactions (Volterra): 1 = do not model.
%   Volterra series would capture nonlinear interactions between
%   conditions; with only one condition, this is not applicable.

% --- Global normalization ---
job{1}.spm.stats.fmri_spec.global                  = 'None';
%   Global normalization: 'None'.  Global scaling can remove true
%   activations that covary with global signal.  We rely on the
%   WM/CSF regressors for physiological noise removal instead.

% --- Masking ---
if nargin < 6 || isempty(mask_file)
    % No explicit mask: SPM uses its default implicit threshold mask
    % (voxels below 0.8 * global mean are excluded).  This provides
    % whole-brain coverage including subcortical structures.
    job{1}.spm.stats.fmri_spec.mthresh              = 0.8;
    job{1}.spm.stats.fmri_spec.mask                 = {''};
else
    % Explicit mask provided (e.g., a grey matter mask).  SPM will
    % restrict estimation to voxels where mask > 0.  This can improve
    % specificity by excluding non-brain voxels but may cause NaN
    % values outside the mask boundary, which is problematic for
    % subcortical ROI analyses.
    job{1}.spm.stats.fmri_spec.mthresh              = 0.8;
    job{1}.spm.stats.fmri_spec.mask                 = {mask_file};
end

% --- Serial correlations ---
job{1}.spm.stats.fmri_spec.cvi                     = 'AR(1)';
%   Autocorrelation model: first-order autoregressive model AR(1).
%   BOLD time series have positive temporal autocorrelation (each
%   timepoint is correlated with the next).  If this structure is
%   ignored, the ordinary least squares (OLS) residual variance
%   estimate is biased downward, inflating t-statistics and producing
%   false positives.
%
%   SPM's AR(1) approach:
%     1. Fit the GLM with OLS to get initial residuals.
%     2. Estimate the AR(1) coefficient (rho) at each voxel from
%        the residual autocorrelation.
%     3. Pool the AR(1) estimates across voxels using ReML
%        (Restricted Maximum Likelihood) to get a global estimate.
%     4. Construct a whitening matrix W from the estimated
%        autocorrelation structure.
%     5. Pre-multiply both the data (Y) and the design matrix (X) by
%        W, then re-estimate betas with OLS on the whitened data.
%   The resulting standard errors and t-statistics are corrected for
%   temporal autocorrelation.
%
%   Reference: Friston KJ et al. (2007) Ch. 8: "Modelling serial
%   correlations." In: Statistical Parametric Mapping.

% =====================================================================
%   MODULE 2: MODEL ESTIMATION (spm.stats.fmri_est)
% =====================================================================
% After the design matrix is constructed, this module fits the GLM.
% It estimates beta weights at every voxel using Restricted Maximum
% Likelihood (ReML) for the variance components and weighted least
% squares for the betas.

job{2}.spm.stats.fmri_est.spmmat(1)                = ...
    cfg_dep('fMRI model specification: SPM.mat File', ...
            substruct('.', 'val', '{}', {1}, ...
                      '.', 'val', '{}', {1}, ...
                      '.', 'val', '{}', {1}), ...
            substruct('.', 'spmmat'));
%   This creates a dependency: the estimation module reads the SPM.mat
%   file produced by the specification module.  SPM's batch system
%   resolves this dependency at runtime, so the two modules can be
%   specified together and run sequentially in a single batch call.

job{2}.spm.stats.fmri_est.write_residuals           = 0;
%   Do not write residual images to disk (saves storage; residuals
%   are rarely needed for standard analyses).

job{2}.spm.stats.fmri_est.method.Classical           = 1;
%   Estimation method: Classical (ReML).  The alternative would be
%   Bayesian estimation, which is not used here.

end
