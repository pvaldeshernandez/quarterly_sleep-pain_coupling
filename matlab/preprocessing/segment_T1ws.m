% SEGMENT_T1WS  Unified segmentation of T1-weighted anatomical images.
%
% =========================================================================
% PIPELINE POSITION: Step 1 (Group A — structural stream)
% =========================================================================
%
% PURPOSE
% -------
% Segment each T1w image into tissue classes (GM, WM, CSF, bone, soft
% tissue, air/background) using SPM12's unified segmentation algorithm
% (Ashburner & Friston, 2005). This step simultaneously estimates tissue
% probability maps, a bias field correction, and an initial affine +
% nonlinear spatial normalization to MNI space. The native-space tissue
% segmentations (c1-c3) and Procrustes-aligned "imported" segmentations
% (rc1-rc3) are required for DARTEL template construction in the next step.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Tissue probability maps : SPM12 default TPMs (TPM.nii, 6 classes)
%   Gaussians per class     : [1 1 2 3 4 2] (GM WM CSF bone soft air)
%   Bias regularization     : 0.001 (light; appropriate for 3T)
%   Bias FWHM               : 60 mm
%   Sampling distance        : 3 mm (controls speed vs. accuracy trade-off)
%   MRF cleanup             : 1 (on) — removes isolated misclassified voxels
%   Affine regularization   : 'mni' (ICBM European template)
%   Warping regularization  : [0 0.001 0.5 0.05 0.2] (SPM12 defaults)
%   Write deformation fields: [1 1] (both forward and inverse)
%
%   Native-space outputs    : c1-c5 (tissue classes 1-5)
%                             rc1-rc3 (Procrustes-aligned, for DARTEL)
%   Warped outputs          : wc1, wc2 (MNI-space GM and WM)
%                             mwc3 (modulated MNI-space CSF)
%
% EXPECTED INPUTS
% ---------------
%   T1w NIfTI files: <folder>/ses-<ses>/anat/sub-*_ses-*_T1w.nii
%
% OUTPUTS
% -------
%   Per subject (written to same anat/ directory):
%     c1*.nii ... c5*.nii   — Native-space tissue probability maps
%     rc1*.nii ... rc3*.nii — DARTEL-imported (Procrustes-aligned) maps
%     wc1*.nii, wc2*.nii    — MNI-warped GM and WM
%     y_*.nii, iy_*.nii     — Forward and inverse deformation fields
%     *_seg8.mat             — Segmentation parameters
%
% REFERENCES
% ----------
%   Ashburner J, Friston KJ. Unified segmentation. NeuroImage.
%   2005;26(3):839-851.
%
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, account, burst, mem, time, user, matlabver, slurmfolder
%
% =========================================================================

% Specify T1ws to be segmented.
filedir = dir(strrep([folder 'ses-' ses '/anat/sub-*_ses-' ses '*_T1w.nii'],'**','*'));

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
    T1w = fullfile(filedir(i).folder,filedir(i).name);
    matlabbatch_all{i}.spm.spatial.preproc.channel.vols = {T1w};
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