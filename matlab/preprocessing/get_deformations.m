% GET_DEFORMATIONS  Compute DARTEL flow fields for non-session-01 images.
%
% =========================================================================
% PIPELINE POSITION: Step 7 (Group C — normalization)
% =========================================================================
%
% PURPOSE
% -------
% The DARTEL template was built from session-01 T1w images only
% (create_T1w_template.m), which also produced per-subject flow fields
% for those images. For images from other sessions or modalities (other-
% session T1ws, fMRI-derived segmentations), new flow fields must be
% estimated by registering their Procrustes-aligned segmentations (rc1,
% rc2, rc3) to the existing template series.
%
% This script uses SPM's "Run Existing Template" DARTEL module
% (spm.tools.dartel.warp1), which holds the template fixed and estimates
% only the individual deformation. The same 6-iteration schedule as the
% original template creation is used to ensure consistency.
%
% DARTEL PARAMETERS (matching manuscript)
% ----------------------------------------
%   Same iteration schedule as create_T1w_template.m:
%     Iter 1: rparam=[4 2 1e-6],     K=0, template=_0
%     Iter 2: rparam=[2 1 1e-6],     K=0, template=_1
%     Iter 3: rparam=[1 0.5 1e-6],   K=1, template=_2
%     Iter 4: rparam=[0.5 0.25 1e-6], K=2, template=_3
%     Iter 5: rparam=[0.25 0.125 1e-6], K=4, template=_4
%     Iter 6: rparam=[0.25 0.125 1e-6], K=6, template=_5
%
%   Optimization:
%     lmreg = 0.01, cyc = 3, its = 3
%
% EXPECTED INPUTS
% ---------------
%   Procrustes-aligned segmentations from non-session-01 T1ws:
%     <folder>/ses-<ses>/anat/rc1sub-*_ses-*_T1w.nii
%     <folder>/ses-<ses>/anat/rc2sub-*_ses-*_T1w.nii
%     <folder>/ses-<ses>/anat/rc3sub-*_ses-*_T1w.nii
%   Procrustes-aligned segmentations from mean functional images:
%     <folder>/ses-<ses>/func/<task>/rc1meanuasub-*_ses-*_bold.nii
%   DARTEL template series from session-01:
%     <folder>/ses-01/anat/<study>_T1w__0.nii through _5.nii
%
% OUTPUTS
% -------
%   Per-subject flow fields:
%     u_rc1sub-*_ses-*_T1w.nii      (for T1w-based deformations)
%     u_rc1meanuasub-*_bold.nii     (for fMRI-based "direct" deformations)
%
% REFERENCES
% ----------
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, study, account, burst, mem, time, user, matlabver,
%   slurmfolder
%
% =========================================================================

% Specify Procrustes-aligned segmentations of other sessions than 1.
rc1s = [
        filedir2cell(dir(strrep([folder 'ses-' ses '/anat/rc1sub-*_ses-' ses '*_T1w.nii'],'**','*')))
        filedir2cell(dir(strrep([folder 'ses-' ses '/func/*/rc1meanuasub-*_ses-' ses '*_bold.nii'],'**','*')))
        ];
rc2s = [filedir2cell(dir(strrep([folder 'ses-' ses '/anat/rc2sub-*_ses-' ses '*_T1w.nii'],'**','*')))
        filedir2cell(dir(strrep([folder 'ses-' ses '/func/*/rc1meanuasub-*_ses-' ses '*_bold.nii'],'**','*')))
        ];
rc3s = [
        filedir2cell(dir(strrep([folder 'ses-' ses '/anat/rc3sub-*_ses-' ses '*_T1w.nii'],'**','*')))
        filedir2cell(dir(strrep([folder 'ses-' ses '/func/*/rc1meanuasub-*_ses-' ses '*_bold.nii'],'**','*')))
        ];

% Specify templates
template0 = fullfile(folder,'ses-01','anat',[study '_T1w__0.nii']);
template1 = fullfile(folder,'ses-01','anat',[study '_T1w__1.nii']);
template2 = fullfile(folder,'ses-01','anat',[study '_T1w__2.nii']);
template3 = fullfile(folder,'ses-01','anat',[study '_T1w__3.nii']);
template4 = fullfile(folder,'ses-01','anat',[study '_T1w__4.nii']);
template5 = fullfile(folder,'ses-01','anat',[study '_T1w__5.nii']);

% Create the batches and store them in a cell.
N = numel(rc1s);
matlabbatch_all = cell(N,1);
for i = 1:N
    matlabbatch_all{i}.spm.tools.dartel.warp1.images = {rc1s(i), rc2s(i), rc3s(i)};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.rform = 0;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(1).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(1).rparam = [4 2 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(1).K = 0;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(1).template = {template0};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(2).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(2).rparam = [2 1 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(2).K = 0;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(2).template = {template1};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(3).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(3).rparam = [1 0.5 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(3).K = 1;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(3).template = {template2};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(4).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(4).rparam = [0.5 0.25 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(4).K = 2;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(4).template = {template3};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(5).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(5).rparam = [0.25 0.125 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(5).K = 4;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(5).template = {template4};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(6).its = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(6).rparam = [0.25 0.125 1e-06];
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(6).K = 6;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.param(6).template = {template5};
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.optim.lmreg = 0.01;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.optim.cyc = 3;
    matlabbatch_all{i}.spm.tools.dartel.warp1.settings.optim.its = 3;
end

% Initialize the cells containing the files needed to run the batches in
% the SLURM.
N = numel(matlabbatch_all);
matoutfiles = cell(N,1);
shfiles = cell(N,1);
logfiles = cell(N,1);
matinfiles = cell(N,1);
strrun = 'run';
conf = struct('account',account,'burst',burst,'mem',mem,'time',time,'user',user,'matlabver',matlabver);
% Send the batches to SLURM.
for i = 1:N
    matlabbatch = matlabbatch_all(i);
    [matoutfiles{i},shfiles{i},logfiles{i},matinfiles{i}] = sendslurm('','spm_jobman',slurmfolder,conf,strrun,matlabbatch);
end
% Clean the files needed to run the batches in the SLURM.
chckdelslurm(matoutfiles,shfiles,logfiles,matinfiles,slurmfolder,1);