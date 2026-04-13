function run_mode(mask_type, study_dir, output_base, subjects, gm_mask_pattern)
% RUN_MODE  Estimate first-level GLMs for all subjects with a given mask mode.
%
%   run_mode(mask_type)
%   run_mode(mask_type, study_dir, output_base, subjects, gm_mask_pattern)
%
%   Called by run_first_level.m to run the SPM12 first-level GLM estimation
%   for one masking mode. Loops over subjects, builds the batch via
%   create_first_level_job, runs SPM, then adds the Stim_vs_Baseline
%   contrast via create_stim_vs_baseline_contrast.
%
% =========================================================================
%   INPUTS
% =========================================================================
%   mask_type       - 'gm'   : restrict estimation to each subject's grey
%                              matter mask (standard cortical analysis)
%                    'none'  : no explicit mask; SPM implicit threshold
%                              (0.8 * global mean) gives full-brain coverage
%                              including subcortical structures
%
%   study_dir       - (optional) Root BIDS directory. Default: see below.
%   output_base     - (optional) Base output directory. A suffix is appended
%                     automatically: '_gm-masked' or '_unmasked'.
%   subjects        - (optional) Cell array of subject IDs. Default: {}.
%   gm_mask_pattern - (optional) Filename pattern for GM mask. Default:
%                     'sUPLOAD2_T1w__gm-2mm-binarized(0.25).nii'
%
% =========================================================================
%   WHY TWO MODES
% =========================================================================
%   GM-masked ('gm'):
%     Restricts estimation to voxels inside each subject's individual GM
%     mask. Standard for cortical analyses. PROBLEM: subcortical ROIs
%     (NAcc, thalamus, brainstem nuclei) lose voxels or become all-NaN.
%
%   Unmasked ('none'):
%     SPM uses its default implicit threshold (0.8 * global mean), giving
%     full-brain coverage. Required for subcortical ROI extraction.
%     TRADE-OFF: more voxels estimated, but WM/CSF nuisance regressors
%     already account for non-grey-matter noise.
%
%   Both versions are needed:
%     - GM-masked  → derivatives/step5_fmri_contrasts_masked/
%     - Unmasked   → derivatives/step5_fmri_contrasts_unmasked/
%   (Python step5 replicates this logic from the stored SPM.mat files.)
%
% -------------------------------------------------------------------------
%   Part of: quarterly_sleep-pain_coupling
%   See also: run_first_level, create_first_level_job,
%             create_stim_vs_baseline_contrast
% -------------------------------------------------------------------------

%% ========================================================================
%   DEFAULTS
%  ========================================================================

if nargin < 2 || isempty(study_dir)
    study_dir = '/orange/cruzalmeida/pvaldeshernandez/Data/UPLOAD2/CONN2SPM_dartel_BIDS_indirect';
end

if nargin < 3 || isempty(output_base)
    output_base = fullfile(study_dir, 'GLM-first_level');
end

if nargin < 4 || isempty(subjects)
    subjects = {};
end

if nargin < 5 || isempty(gm_mask_pattern)
    gm_mask_pattern = 'sUPLOAD2_T1w__gm-2mm-binarized(0.25).nii';
end

%% ========================================================================
%   VALIDATION
%  ========================================================================

switch lower(mask_type)
    case 'gm'
        output_suffix = '_gm-masked';
    case 'none'
        output_suffix = '_unmasked';
    otherwise
        error('run_mode:BadMaskType', ...
              'mask_type must be ''gm'' or ''none''. Got: ''%s''', mask_type);
end

%% ========================================================================
%   INITIALIZATION
%  ========================================================================

spm('defaults', 'FMRI');
spm_jobman('initcfg');

% Auto-detect subjects if list is empty
if isempty(subjects)
    d = dir(study_dir);
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    subjects = {d.name};
    has_func = false(size(subjects));
    for i = 1:numel(subjects)
        if exist(fullfile(study_dir, subjects{i}, 'func'), 'dir')
            has_func(i) = true;
        end
    end
    subjects = subjects(has_func);
    fprintf('Auto-detected %d subjects with functional data.\n', numel(subjects));
end

output_base_full = [output_base output_suffix];
if ~exist(output_base_full, 'dir')
    mkdir(output_base_full);
end

%% ========================================================================
%   MAIN LOOP
%  ========================================================================

n_subjects = numel(subjects);
fprintf('\n========================================\n');
fprintf('  First-level GLM: mask_type = %s\n', mask_type);
fprintf('  Subjects: %d\n', n_subjects);
fprintf('  Output:   %s\n', output_base_full);
fprintf('========================================\n\n');

n_success = 0; n_skip = 0; n_fail = 0;
tic;

for s = 1:n_subjects
    subj_id    = subjects{s};
    func_dir   = fullfile(study_dir, subj_id, 'func');
    output_dir = fullfile(output_base_full, subj_id);

    fprintf('[%3d/%3d] %s ... ', s, n_subjects, subj_id);

    if exist(fullfile(output_dir, 'SPM.mat'), 'file')
        fprintf('already done, skipping.\n');
        n_skip = n_skip + 1;
        continue;
    end

    try
        % Motion parameters
        motion_files = dir(fullfile(func_dir, 'rp_*.txt'));
        if isempty(motion_files)
            error('No motion file (rp_*.txt) in %s', func_dir);
        end
        motion_file = fullfile(func_dir, motion_files(1).name);

        % WM / CSF timecourses
        wm_file  = dir(fullfile(func_dir, '*_WM_signal.txt'));
        csf_file = dir(fullfile(func_dir, '*_CSF_signal.txt'));
        if isempty(wm_file),  wm_file  = dir(fullfile(func_dir, 'WM_timecourse.txt'));  end
        if isempty(csf_file), csf_file = dir(fullfile(func_dir, 'CSF_timecourse.txt')); end
        if isempty(wm_file) || isempty(csf_file)
            error('WM or CSF timecourse not found in %s', func_dir);
        end
        wm_signal  = load(fullfile(func_dir, wm_file(1).name));
        csf_signal = load(fullfile(func_dir, csf_file(1).name));

        % GM mask (only for 'gm' mode)
        switch lower(mask_type)
            case 'gm'
                anat_dir = fullfile(study_dir, subj_id, 'anat');
                mask_candidates = dir(fullfile(anat_dir, gm_mask_pattern));
                if isempty(mask_candidates)
                    error('GM mask (%s) not found in %s', gm_mask_pattern, anat_dir);
                end
                mask_file = fullfile(anat_dir, mask_candidates(1).name);
            case 'none'
                mask_file = '';
        end

        % Build and run GLM batch
        job = create_first_level_job(func_dir, output_dir, ...
                                     motion_file, wm_signal, csf_signal, ...
                                     mask_file);
        spm_jobman('run', job);

        % Add Stim_vs_Baseline contrast
        create_stim_vs_baseline_contrast(fullfile(output_dir, 'SPM.mat'));

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
fprintf('  Mode: %s — complete\n', mask_type);
fprintf('  Succeeded: %d  |  Skipped: %d  |  Failed: %d\n', ...
        n_success, n_skip, n_fail);
fprintf('  Total time: %.1f min\n', elapsed / 60);
fprintf('========================================\n');

if n_fail > 0
    fprintf('WARNING: %d subject(s) failed.\n', n_fail);
end

end
