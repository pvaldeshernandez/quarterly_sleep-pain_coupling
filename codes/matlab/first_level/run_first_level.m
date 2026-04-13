% RUN_FIRST_LEVEL  Estimate first-level GLMs with and without GM mask.
%
%   This is the top-level script for first-level fMRI analysis. It runs
%   the SPM12 GLM estimation twice for all subjects:
%
%     1. GM-masked ('gm')   → output_base_gm-masked/
%        Standard cortical analysis. Each subject's individual grey matter
%        mask restricts estimation to GM voxels. Used for cortical ROIs
%        (S1, insula, NAcc with sufficient GM coverage).
%
%     2. Unmasked ('none')  → output_base_unmasked/
%        Full-brain estimation using SPM's implicit threshold (0.8 * global
%        mean). Required for subcortical ROIs (thalamus, brainstem nuclei)
%        where individual GM masks cause voxel dropout or all-NaN contrasts.
%
%   Both runs use the same design matrix, nuisance regressors, HRF, and
%   contrast vector. The only difference is the spatial extent of the output.
%
%   NOTE: This MATLAB pipeline produces the SPM.mat files and contrast
%   images stored in data/original/spm_mats/. The Python step5
%   (step5_estimate_fmri_contrasts.py) replicates the unmasked re-estimation
%   directly from the stored SPM.mat files, so Xiaohan does not need to run
%   this script.
%
% =========================================================================
%   USER-CONFIGURABLE PARAMETERS
% =========================================================================
%   Edit these variables to match your local directory structure.

% Root BIDS-like directory containing subject subdirectories
study_dir = '/orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/CONN2SPM_dartel_BIDS_indirect';

% Base output directory (a suffix is appended per mode: _gm-masked, _unmasked)
output_base = fullfile(study_dir, 'GLM-first_level');

% Subject IDs — leave empty to auto-detect from study_dir
subjects = {};

% GM mask filename pattern (only used in 'gm' mode)
gm_mask_pattern = 'sUPLOAD2_T1w__gm-2mm-binarized(0.25).nii';

% =========================================================================
%   RUN BOTH MODES
% =========================================================================

fprintf('\n============================================================\n');
fprintf('  First-level GLM estimation — both masking modes\n');
fprintf('============================================================\n\n');

fprintf('>>> Mode 1 of 2: GM-masked\n\n');
run_mode('gm', study_dir, output_base, subjects, gm_mask_pattern);

fprintf('\n>>> Mode 2 of 2: Unmasked\n\n');
run_mode('none', study_dir, output_base, subjects, gm_mask_pattern);

fprintf('\n============================================================\n');
fprintf('  Both modes complete.\n');
fprintf('============================================================\n');
