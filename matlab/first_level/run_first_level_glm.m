% RUN_FIRST_LEVEL_GLM  Estimate first-level GLMs for all subjects.
%
%   This script loops over all subjects in the study, constructs a
%   first-level GLM batch for each one (via create_first_level_job),
%   and runs the batch through SPM's job manager.
%
% =========================================================================
%   WHY TWO ESTIMATION MODES (MASKED vs. UNMASKED)
% =========================================================================
%   The mask_type parameter controls whether an explicit grey matter (GM)
%   mask is applied during GLM estimation:
%
%   'gm' (masked):
%     SPM restricts estimation to voxels inside the subject's GM mask.
%     This is the standard approach for cortical analyses because it
%     reduces the number of voxels tested (improving multiple-comparison
%     correction) and ensures that only grey matter voxels contribute to
%     the results.
%
%     PROBLEM: GM masks are derived from each subject's T1 segmentation.
%     They follow individual cortical anatomy and may exclude subcortical
%     structures (nucleus accumbens, thalamus, brainstem nuclei) or voxels
%     at tissue boundaries.  This creates NaN values outside each subject's
%     GM boundary in the contrast images.  When extracting ROI values from
%     these contrast images, subcortical ROIs can lose a significant
%     fraction of their voxels.  For example, a 10-voxel left thalamus ROI
%     may retain only 2 voxels after masking, and a parabrachial nucleus
%     (PBN) ROI at fMRI resolution may have ZERO valid voxels.
%
%   'none' (unmasked):
%     No explicit GM mask is applied.  SPM uses its default implicit mask
%     (threshold = 0.8 * global mean), which provides whole-brain coverage
%     including all subcortical structures.  The contrast images have valid
%     values everywhere inside the brain, enabling complete ROI coverage.
%
%     TRADE-OFF: More voxels are estimated (including white matter and
%     some CSF-adjacent voxels), but this is acceptable because:
%       (a) We are not performing whole-brain voxel-wise inference.
%       (b) We extract mean ROI values, which average over multiple voxels
%           and are robust to noise in individual voxels.
%       (c) The WM and CSF nuisance regressors already account for
%           physiological noise in non-grey-matter tissue.
%
%   In this study, we run BOTH versions:
%     - GM-masked contrasts for cortical ROIs (e.g., insula, S1)
%     - Unmasked contrasts for subcortical ROIs (e.g., NAcc, thalamus,
%       brainstem arousal nuclei)
%
%   See also: create_first_level_job, create_stim_vs_baseline_contrast
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
% -------------------------------------------------------------------------

%% ========================================================================
%   USER-CONFIGURABLE PARAMETERS
%  ========================================================================
% Edit the variables in this section to match your local directory
% structure and subject list.

% --- Root directory of the BIDS-like study ---
% This should contain subject-level subdirectories, each with a 'func'
% folder holding preprocessed functional data and an 'anat' folder
% holding structural segmentation outputs.
study_dir = '/orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/CONN2SPM_dartel_BIDS_indirect';

% --- Subject IDs ---
% Cell array of subject identifiers.  Each entry should match a
% subdirectory name under study_dir.  If left empty, the script will
% automatically detect all subdirectories that contain an SPM-compatible
% functional directory.
subjects = {};  % e.g., {'sub-001', 'sub-002', ...}
%   Tip: to auto-detect subjects from the study directory, leave this
%   empty and the script will populate it (see below).

% --- Mask type: 'gm' or 'none' ---
% Controls whether an explicit grey matter mask is applied.
% See the header documentation above for a detailed explanation.
mask_type = 'none';  % 'gm' for grey matter mask, 'none' for unmasked

% --- Output base directory ---
% First-level SPM.mat and beta images will be written to:
%   <output_base>/<subject_id>/
% A suffix is appended based on mask_type to keep the two versions
% separate on disk.
output_base = fullfile(study_dir, 'GLM-first_level');

% --- GM mask filename pattern ---
% Only used when mask_type = 'gm'.  This is the filename of the grey
% matter mask image, expected to reside in each subject's anat directory.
% Typical names: 'wc1*.nii' (normalized GM from segmentation) or
% 'brain_mask.nii' (binarized GM mask).
gm_mask_pattern = 'wc1*.nii';

%% ========================================================================
%   INITIALIZATION
%  ========================================================================

% Start SPM in batch mode (no GUI windows)
spm('defaults', 'FMRI');
spm_jobman('initcfg');
%   spm('defaults', 'FMRI') loads the default settings for fMRI analysis.
%   spm_jobman('initcfg') initializes the batch system, which is required
%   before running any batch jobs programmatically.

% Auto-detect subjects if the list is empty
if isempty(subjects)
    d = dir(study_dir);
    d = d([d.isdir]);                         % keep only directories
    d = d(~ismember({d.name}, {'.', '..'}));  % remove . and ..
    subjects = {d.name};

    % Filter to subjects that actually have functional data
    has_func = false(size(subjects));
    for i = 1:numel(subjects)
        func_dir_candidate = fullfile(study_dir, subjects{i}, 'func');
        if exist(func_dir_candidate, 'dir')
            has_func(i) = true;
        end
    end
    subjects = subjects(has_func);

    fprintf('Auto-detected %d subjects with functional data.\n', numel(subjects));
end

% Append mask-type suffix to output directory
switch lower(mask_type)
    case 'gm'
        output_base_full = [output_base '_gm-masked'];
    case 'none'
        output_base_full = [output_base '_unmasked'];
    otherwise
        error('run_first_level_glm:BadMaskType', ...
              'mask_type must be ''gm'' or ''none''. Got: ''%s''', mask_type);
end

% Create the output base directory if it does not exist
if ~exist(output_base_full, 'dir')
    mkdir(output_base_full);
end

%% ========================================================================
%   MAIN LOOP: PROCESS EACH SUBJECT
%  ========================================================================

n_subjects = numel(subjects);
fprintf('\n========================================\n');
fprintf('  First-level GLM estimation\n');
fprintf('  Subjects:  %d\n', n_subjects);
fprintf('  Mask type: %s\n', mask_type);
fprintf('  Output:    %s\n', output_base_full);
fprintf('========================================\n\n');

n_success = 0;
n_skip    = 0;
n_fail    = 0;
tic;

for s = 1:n_subjects
    subj_id = subjects{s};
    fprintf('[%3d/%3d] %s ... ', s, n_subjects, subj_id);

    % --- Define paths for this subject ---
    func_dir   = fullfile(study_dir, subj_id, 'func');
    output_dir = fullfile(output_base_full, subj_id);

    % Skip if SPM.mat already exists (estimation already done)
    if exist(fullfile(output_dir, 'SPM.mat'), 'file')
        fprintf('already done (SPM.mat exists), skipping.\n');
        n_skip = n_skip + 1;
        continue;
    end

    try
        % --- Locate the motion parameter file ---
        % SPM's realignment produces a text file named rp_<funcname>.txt
        % in the functional directory.
        motion_files = dir(fullfile(func_dir, 'rp_*.txt'));
        if isempty(motion_files)
            error('No motion parameter file (rp_*.txt) found in %s', func_dir);
        end
        motion_file = fullfile(func_dir, motion_files(1).name);

        % --- Load WM and CSF timecourses ---
        % These are expected as text files in the functional directory,
        % produced by CONN or a custom extraction script.  Filenames may
        % vary; adjust the patterns below to match your data.
        wm_file  = dir(fullfile(func_dir, '*_WM_signal.txt'));
        csf_file = dir(fullfile(func_dir, '*_CSF_signal.txt'));

        if isempty(wm_file)
            % Alternative: CONN-style naming
            wm_file = dir(fullfile(func_dir, 'WM_timecourse.txt'));
        end
        if isempty(csf_file)
            csf_file = dir(fullfile(func_dir, 'CSF_timecourse.txt'));
        end

        if isempty(wm_file) || isempty(csf_file)
            error('WM or CSF timecourse file not found in %s', func_dir);
        end

        wm_signal  = load(fullfile(func_dir, wm_file(1).name));
        csf_signal = load(fullfile(func_dir, csf_file(1).name));

        % --- Determine the mask file ---
        switch lower(mask_type)
            case 'gm'
                % Look for the GM mask in the subject's anat directory
                anat_dir = fullfile(study_dir, subj_id, 'anat');
                mask_candidates = dir(fullfile(anat_dir, gm_mask_pattern));
                if isempty(mask_candidates)
                    error('GM mask (%s) not found in %s', ...
                          gm_mask_pattern, anat_dir);
                end
                mask_file = fullfile(anat_dir, mask_candidates(1).name);
            case 'none'
                mask_file = '';
        end

        % --- Build and run the batch ---
        job = create_first_level_job(func_dir, output_dir, ...
                                     motion_file, wm_signal, csf_signal, ...
                                     mask_file);
        spm_jobman('run', job);

        fprintf('done.\n');
        n_success = n_success + 1;

    catch ME
        fprintf('FAILED: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
end

%% ========================================================================
%   SUMMARY
%  ========================================================================

elapsed = toc;
fprintf('\n========================================\n');
fprintf('  First-level GLM estimation complete\n');
fprintf('  Succeeded:  %d\n', n_success);
fprintf('  Skipped:    %d\n', n_skip);
fprintf('  Failed:     %d\n', n_fail);
fprintf('  Total time: %.1f minutes (%.1f s/subject)\n', ...
        elapsed / 60, elapsed / max(n_success, 1));
fprintf('========================================\n');

% If any subjects failed, print a warning
if n_fail > 0
    fprintf('\nWARNING: %d subject(s) failed. Review the error messages above.\n', n_fail);
end
