% REALIGN_UNWARP  Realign functional volumes and correct for susceptibility-
%                 by-movement interactions.
%
% =========================================================================
% PIPELINE POSITION: Step 4 (Group B — functional stream)
% =========================================================================
%
% PURPOSE
% -------
% Head motion during fMRI acquisition causes two problems: (1) spatial
% misalignment between volumes, and (2) changes in susceptibility-induced
% distortions that vary with head position. Standard realignment corrects
% only the first problem. Realign & Unwarp (Andersson et al., 2001)
% additionally models and removes movement-related signal changes caused
% by the interaction between head motion and the static inhomogeneity
% field. This is especially important near air-tissue interfaces (e.g.,
% orbitofrontal cortex, temporal poles) relevant to pain processing.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Estimation:
%     Quality         : 1.0 (highest; uses all voxels for cost function)
%     Separation      : 3 mm (sampling distance for registration)
%     Smoothing FWHM  : 5 mm (applied to images before cost evaluation)
%     Register to     : First image (rtm=0)
%     Interpolation   : 7th-degree B-spline (einterp=7; highest SPM option,
%                       minimizes interpolation artifacts)
%     Wrapping        : [0 0 0] (no wrap-around in any dimension)
%     Weighting       : none
%
%   Unwarp estimation:
%     Basis functions  : [12 12] (DCT basis set for field modeling)
%     Regularization   : 1st order
%     Lambda           : 100000 (regularization weight)
%     Jacobian modulation: 0 (off)
%     First-order terms: [4 5] (pitch and roll; the two rotation axes
%                        with the largest susceptibility-by-movement
%                        interaction effects)
%     Second-order terms: [] (none)
%     Smoothing FWHM   : 4 mm
%     Re-estimation     : 1 (re-estimate movement after unwarp)
%     Number of iterations: 5
%     Expansion point   : 'Average'
%
%   Reslice:
%     Which images     : [2 1] (all images + mean image)
%     Interpolation    : 4th-degree B-spline
%     Wrapping         : [0 0 0]
%     Masking          : 1 (mask images to remove edge artifacts)
%     Prefix           : 'u'
%
% EXPECTED INPUTS
% ---------------
%   Slice-timing corrected BOLD files (prefix 'a'):
%     <folder>/ses-<ses>/func/<task>/asub-*_ses-*_task-*_bold.nii
%
% OUTPUTS
% -------
%   Realigned + unwarped images (prefix 'ua'):
%     <folder>/ses-<ses>/func/<task>/uasub-*_ses-*_task-*_bold.nii
%   Mean realigned + unwarped image:
%     <folder>/ses-<ses>/func/<task>/meanuasub-*_ses-*_task-*_bold.nii
%   Realignment parameters:
%     <folder>/ses-<ses>/func/<task>/rp_asub-*_ses-*_task-*_bold.txt
%
% REFERENCES
% ----------
%   Andersson JLR, Hutton C, Ashburner J, Turner R, Friston K. Modeling
%   geometric deformations in EPI time series. NeuroImage.
%   2001;13(5):903-919.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, account, burst, mem, time, user, matlabver,
%   slurmfolder
%
% =========================================================================

% Specify the fMRI files to be realigned and unwarpped (prefix 'a').
filedir = dir(strrep([folder 'ses-' ses '/func/' task '/asub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Create the batches and store them in a cell.
N = numel(filedir);
matlabbatch_all = cell(N,1);
for i = 1:N
    fMRI = fullfile(filedir(i).folder,filedir(i).name);
    matlabbatch_all{i}.spm.spatial.realignunwarp.data.scans = {fMRI};
    matlabbatch_all{i}.spm.spatial.realignunwarp.data.pmscan = '';
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.quality = 1;
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.sep = 3;
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.rtm = 0;
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.einterp = 7;
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
    matlabbatch_all{i}.spm.spatial.realignunwarp.eoptions.weight = '';
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.jm = 0;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.sot = [];
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.rem = 1;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.noi = 5;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
    matlabbatch_all{i}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
    matlabbatch_all{i}.spm.spatial.realignunwarp.uwroptions.rinterp = 4;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
    matlabbatch_all{i}.spm.spatial.realignunwarp.uwroptions.mask = 1;
    matlabbatch_all{i}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';
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