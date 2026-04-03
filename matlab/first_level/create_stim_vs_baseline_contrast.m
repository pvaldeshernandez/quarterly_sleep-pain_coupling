function create_stim_vs_baseline_contrast(spm_mat_path)
% CREATE_STIM_VS_BASELINE_CONTRAST  Add a T-contrast to an estimated GLM.
%
%   create_stim_vs_baseline_contrast(spm_mat_path)
%
%   This function adds a T-contrast named 'Stim_vs_Baseline' to an
%   existing first-level SPM.mat file that has already been estimated.
%   It then runs the contrast manager to produce the corresponding
%   con_0001.nii and spmT_0001.nii images.
%
% =========================================================================
%   WHAT THIS CONTRAST MEANS
% =========================================================================
%   The first-level GLM for this study has a single task regressor: the
%   five stimulation blocks convolved with the canonical hemodynamic
%   response function (HRF).  The remaining columns of the design matrix
%   are nuisance regressors (motion parameters, WM signal, CSF signal)
%   and the high-pass filter constants (DCT basis set, added implicitly
%   by SPM).
%
%   The T-contrast vector [1] tests whether the task regressor has a
%   positive effect at each voxel.  In SPM's GLM framework, a T-contrast
%   with a single weight on the first column is equivalent to:
%
%       H0: beta_stimulation = 0    (no activation during stimulation)
%       H1: beta_stimulation > 0    (activation during stimulation)
%
%   The "baseline" against which stimulation is compared is IMPLICIT: it
%   consists of all time points NOT modeled by the task regressor.  In a
%   block design, these are the rest periods between stimulation blocks
%   and any post-stimulation recovery periods.  Because the HRF
%   convolution spreads the modeled response slightly beyond the 24 s
%   block boundaries, the effective baseline is the sustained signal
%   level during prolonged rest intervals.
%
%   Key point: the contrast [1] does NOT compare stimulation to a
%   separate "rest" condition.  There is no explicit rest regressor.
%   Instead, the intercept (constant term, added automatically by SPM)
%   captures the mean signal level.  The task regressor captures the
%   DEVIATION from that mean during stimulation blocks.  A positive
%   beta_stimulation therefore means the voxel's signal is higher during
%   stimulation than during the average of all other time points.
%
%   This is the standard approach for single-condition block designs
%   and is sometimes called "stimulation versus implicit baseline."
%
% =========================================================================
%   CONTRAST VECTOR DETAILS
% =========================================================================
%   The contrast vector is [1], which SPM automatically zero-pads to
%   match the total number of regressors in the design matrix.  The
%   full design matrix for this GLM has the following column order:
%
%     Column  1:  Stimulation (task regressor, convolved with HRF)
%     Columns 2-7:  Six motion parameters (x, y, z, pitch, roll, yaw)
%     Column  8:  WM signal (white matter physiological noise)
%     Column  9:  CSF signal (cerebrospinal fluid physiological noise)
%     Column 10+: DCT basis set (high-pass filter, added by SPM)
%     Last column: Constant (session mean, added by SPM)
%
%   The contrast [1] assigns weight 1 to the Stimulation column and
%   (implicitly) weight 0 to everything else.  The nuisance regressors
%   and filter columns are not tested.
%
% =========================================================================
%   INPUT
% =========================================================================
%   spm_mat_path - [string] Full path to an estimated SPM.mat file.
%                  The GLM must have been specified and estimated before
%                  calling this function.  Typically this file resides in
%                  the subject's first-level output directory:
%                  e.g., '/data/first_level/sub-001/SPM.mat'
%
% =========================================================================
%   OUTPUT FILES
% =========================================================================
%   The function writes the following images to the same directory as
%   SPM.mat:
%
%   con_0001.nii  - Contrast image.  Each voxel contains the weighted
%                   sum of the estimated betas according to the contrast
%                   vector.  Here, con = 1 * beta_stimulation.  This
%                   image is used for ROI extraction (extracting the mean
%                   stimulation effect within anatomical regions of
%                   interest).
%
%   spmT_0001.nii - T-statistic image.  Each voxel contains the
%                   t-statistic for the contrast, computed as:
%                     t = c' * beta / sqrt(c' * (X'X)^-1 * c * sigma^2)
%                   where c is the contrast vector, beta is the estimated
%                   parameter vector, X is the (whitened) design matrix,
%                   and sigma^2 is the estimated residual variance.
%                   This image is used for voxel-wise inference (SPM
%                   glass brain maps, cluster-level statistics, etc.).
%
% =========================================================================
%   USAGE EXAMPLE
% =========================================================================
%   % After running create_first_level_job and spm_jobman('run', job):
%   spm_mat_path = '/data/first_level/sub-001/SPM.mat';
%   create_stim_vs_baseline_contrast(spm_mat_path);
%
%   % The contrast and T-stat images are now available:
%   %   /data/first_level/sub-001/con_0001.nii
%   %   /data/first_level/sub-001/spmT_0001.nii
%
% =========================================================================
%   REFERENCE
% =========================================================================
%   Manuscript Methods, "MRI preprocessing and first-level analysis":
%   "The contrast of interest was the T-contrast for the effect of
%   stimulation versus implicit baseline (rest and post-stimulation
%   periods combined)."
%
%   For the theory of contrasts in the GLM:
%   Friston KJ, Ashburner JT, Kiebel SJ, Nichols TE, Penny WD (eds).
%   Statistical Parametric Mapping: The Analysis of Functional Brain
%   Images. Academic Press, 2007. Chapter 9: "Contrasts and classical
%   inference."
%
% -------------------------------------------------------------------------
%   Part of: quarterly_sleep-pain_coupling
%   See also: create_first_level_job, run_first_level_glm
% -------------------------------------------------------------------------

% =====================================================================
%   INPUT VALIDATION
% =====================================================================

if nargin < 1 || isempty(spm_mat_path)
    error('create_stim_vs_baseline_contrast:BadInput', ...
          'You must provide the path to an estimated SPM.mat file.');
end

if ~exist(spm_mat_path, 'file')
    error('create_stim_vs_baseline_contrast:FileNotFound', ...
          'SPM.mat not found: %s', spm_mat_path);
end

% Verify that the GLM has been estimated by checking for the
% existence of at least one beta image in the same directory.
[spm_dir, ~, ~] = fileparts(spm_mat_path);
beta_check = dir(fullfile(spm_dir, 'beta_0001.*'));
if isempty(beta_check)
    error('create_stim_vs_baseline_contrast:NotEstimated', ...
          ['No beta images found in %s. The GLM must be estimated ' ...
           'before contrasts can be defined.'], spm_dir);
end

% =====================================================================
%   BUILD THE CONTRAST MANAGER BATCH
% =====================================================================
% SPM's contrast manager (spm.stats.con) adds contrast definitions to
% an existing SPM.mat and computes the corresponding contrast and
% T-statistic images.

job = {};

% Point to the estimated SPM.mat
job{1}.spm.stats.con.spmmat = {spm_mat_path};

% --- Define the T-contrast ---
job{1}.spm.stats.con.consess{1}.tcon.name    = 'Stim_vs_Baseline';
%   Human-readable name for this contrast.  It will appear in the SPM
%   results interface and in the filename prefix of the output images.

job{1}.spm.stats.con.consess{1}.tcon.weights = 1;
%   Contrast vector: [1].  This is a scalar because we only need to
%   specify the weight for the first regressor (Stimulation).  SPM
%   automatically zero-pads the vector to match the number of columns
%   in the design matrix:
%     [1] -> [1 0 0 0 0 0 0 0 0 ... 0]
%   This tests H0: beta_stimulation = 0 against H1: beta_stimulation > 0.

job{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
%   Session replication: 'none'.  We have a single session per subject,
%   so no replication is needed.  (In multi-session designs, 'repl'
%   would create the contrast separately for each session, and 'replsc'
%   would create a session-averaged contrast.)

% --- Delete existing contrasts ---
job{1}.spm.stats.con.delete = 1;
%   Delete any previously defined contrasts before adding this one.
%   This ensures that con_0001.nii always corresponds to the
%   Stim_vs_Baseline contrast, regardless of how many times the
%   script is run.  Set to 0 if you want to append rather than replace.

% =====================================================================
%   RUN THE CONTRAST MANAGER
% =====================================================================

% Initialize SPM batch system (safe to call multiple times)
spm('defaults', 'FMRI');
spm_jobman('initcfg');

% Execute the batch
fprintf('  Adding contrast ''Stim_vs_Baseline'' to %s\n', spm_mat_path);
spm_jobman('run', job);
fprintf('  Done. Output: con_0001.nii, spmT_0001.nii\n');

end
