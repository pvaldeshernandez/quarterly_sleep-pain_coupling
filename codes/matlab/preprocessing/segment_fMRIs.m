% SEGMENT_FMRIS  Segment mean functional images for DARTEL registration.
%
% =========================================================================
% PIPELINE POSITION: Step 6 (Group B — functional stream)
% =========================================================================
%
% PURPOSE
% -------
% Segment the mean realigned-unwarped functional images using the same
% unified segmentation as the T1ws. This produces Procrustes-aligned
% tissue maps (rc1, rc2, rc3) in the native functional space, which are
% needed by get_deformations.m to compute DARTEL flow fields that can
% warp functional data directly into the study template space.
%
% This "direct" normalization pathway segments the functional images
% themselves, rather than relying solely on the T1w-derived flow fields
% (the "indirect" pathway). Having both options allows quality comparison
% and provides a fallback when T1w-to-functional coregistration is
% imperfect.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Identical to segment_T1ws.m:
%     Tissue probability maps : SPM12 default TPMs (TPM.nii)
%     Gaussians per class     : [1 1 2 3 4 2]
%     Bias regularization     : 0.001
%     Bias FWHM               : 60 mm
%     Sampling distance        : 3 mm
%     MRF cleanup             : 1 (on)
%     Affine regularization   : 'mni'
%     Warping regularization  : [0 0.001 0.5 0.05 0.2]
%     Write deformation fields: [1 1]
%
% EXPECTED INPUTS
% ---------------
%   Mean functional images (after realign+unwarp and coregistration):
%     <folder>/ses-<ses>/func/<task>/meanuasub-*_ses-*_task-*_bold.nii
%
% OUTPUTS
% -------
%   Per subject (written to the func/<task>/ directory):
%     c1meanuasub-*.nii ... c5meanuasub-*.nii  — Native tissue maps
%     rc1meanuasub-*.nii ... rc3meanuasub-*.nii — DARTEL-imported maps
%     y_meanuasub-*.nii, iy_meanuasub-*.nii    — Deformation fields
%     meanuasub-*_seg8.mat                      — Segmentation parameters
%
% REFERENCES
% ----------
%   Ashburner J, Friston KJ. Unified segmentation. NeuroImage.
%   2005;26(3):839-851.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, account, burst, mem, time, user, matlabver,
%   slurmfolder
%
% =========================================================================

% Specify fMRIs to be segmented.
filedir = dir(strrep([folder 'ses-' ses '/func/' task '/meanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Specify segmentation parameters.
spmfolder = fileparts(which('spm.m'));
tpm = fullfile(spmfolder,'tpm','TPM.nii');
ngaus = [1 1 2 3 4 2];
native = [ones(3,2); ones(2,1) zeros(2,1); zeros(1,2)];
warped = [ones(2,2); 0 1; zeros(3,2)];

% Create the batches and store them in a cell.
N = size(filedir,1);
matlabbatch_all = cell(N,1);
for i = 1:N
    fMRI = fullfile(filedir(i).folder,filedir(i).name);
    matlabbatch_all{i}.spm.spatial.preproc.channel.vols = {fMRI};
    matlabbatch_all{i}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch_all{i}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch_all{i}.spm.spatial.preproc.channel.write = [0 0];
    for j = 1:6
        matlabbatch_all{i}.spm.spatial.preproc.tissue(j).tpm = {sprintf('%s,%d',tpm,j)};
        matlabbatch_all{i}.spm.spatial.preproc.tissue(j).ngaus = ngaus(j);
        matlabbatch_all{i}.spm.spatial.preproc.tissue(j).native = native(j,:);
        matlabbatch_all{i}.spm.spatial.preproc.tissue(j).warped = warped(j,:);
    end
    matlabbatch_all{i}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch_all{i}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch_all{i}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch_all{i}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch_all{i}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch_all{i}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch_all{i}.spm.spatial.preproc.warp.write = [1 1];
end

% Initialize the cells containing the files needed to run the batches in
% the SLURM.
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