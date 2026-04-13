% CALCULATE_AVERAGES  Compute group-average normalized images for QC.
%
% =========================================================================
% PIPELINE POSITION: Step 10 (Group C — normalization, final step)
% =========================================================================
%
% PURPOSE
% -------
% After spatial normalization, compute group-mean images of:
%   (1) Normalized T1-weighted anatomical images
%   (2) Normalized mean functional images (indirect pathway, prefix 'd')
%   (3) Normalized mean functional images (direct pathway, prefix 'd2')
%
% These averages serve as visual quality-control tools. A sharp, well-
% defined group average indicates successful inter-subject registration;
% blurring in specific regions may reveal systematic misalignment. The
% average T1w is also used as an underlay for group-level activation
% maps and ROI visualization.
%
% WHAT IT DOES
% ------------
% For each image set, iterates over all subjects, reads each normalized
% volume, accumulates a running sum, then divides by N to produce the
% group average. The result is written as a single NIfTI file in the
% study root folder.
%
% EXPECTED INPUTS
% ---------------
%   Normalized T1ws:
%     <folder>/ses-<ses>/anat/dsub-*_ses-*_T1w.nii
%   Normalized indirect fMRIs:
%     <folder>/ses-<ses>/func/<task>/dmeanuasub-*_bold.nii
%   Normalized direct fMRIs:
%     <folder>/ses-<ses>/func/<task>/d2meanuasub-*_bold.nii
%
% OUTPUTS
% -------
%   <folder>/<study>_averageT1w.nii
%   <folder>/<study>_average_indirect_fMRI.nii
%   <folder>/<study>_average_direct_fMRI.nii
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, study
%
% =========================================================================

%% Collect all spatially normalized T1ws
dT1ws = filedir2cell(dir(strrep([folder 'ses-' ses '/anat/dsub-*_ses-' ses '*_T1w.nii'],'**','*')));

% Calculate and write the average normalized T1w
A = 0;
N = numel(dT1ws);
hwait = waitbar(0,'calculating average dT1w...');
for i = 1:N
    V = spm_vol(dT1ws{i});
    A = A + spm_read_vols(V);
    waitbar(i/N,hwait);
end
close(hwait);
A = A/N;
Va = V;
Va.fname = fullfile(folder,[study '_averageT1w.nii']);
spm_write_vol(Va,A);

%% Collect all spatially normalized indirect fMRIs.
dfMRIs = filedir2cell(dir(strrep([folder 'ses-' ses '/func/' task '/dmeanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*')));

% Calculate and write the average normalized indirect fMRIs.
A = 0;
N = numel(dfMRIs);
hwait = waitbar(0,'calculating average dfMRI...');
for i = 1:N
    V = spm_vol(dfMRIs{i});
    A = A + spm_read_vols(V);
    waitbar(i/N,hwait);
end
close(hwait);
A = A/N;
Va = V;
Va.fname = fullfile(folder,[study '_average_indirect_fMRI.nii']);
spm_write_vol(Va,A);

% Collect all spatially normalized direct fMRIs.
d2fMRIs = filedir2cell(dir(strrep([folder 'ses-' ses '/func/' task '/d2meanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*')));

%% Calculate and write the average normalized direct fMRI.
A = 0;
N = numel(d2fMRIs);
hwait = waitbar(0,'calculating average d2fMRI...');
for i = 1:N
    V = spm_vol(d2fMRIs{i});
    A = A + spm_read_vols(V);
    waitbar(i/N,hwait);
end
close(hwait);
A = A/N;
Va = V;
Va.fname = fullfile(folder,[study '_average_direct_fMRI.nii']);
spm_write_vol(Va,A);