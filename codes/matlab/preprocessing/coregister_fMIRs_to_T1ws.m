% COREGISTER_FMIRS_TO_T1WS  Coregister mean functional images to T1w
%                            anatomical images.
%
% =========================================================================
% PIPELINE POSITION: Step 5 (Group B — functional stream)
% =========================================================================
%
% PURPOSE
% -------
% Functional EPI images and T1-weighted structural images are typically
% acquired with different sequences, slice prescriptions, and geometric
% distortions. Coregistration aligns the mean realigned-unwarped functional
% image to the T1w anatomical image using a rigid-body transformation
% estimated by maximizing normalized mutual information (NMI). This
% alignment is necessary so that the DARTEL deformation fields computed
% from the T1w (structural stream) can be correctly applied to the
% functional data.
%
% The estimated transformation is applied to both the mean functional
% image and the full 4D time series, so all functional data end up in
% the native T1w space.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Cost function      : Normalized Mutual Information (NMI)
%                        Robust to intensity differences between EPI and T1w
%   Separation         : [4 2] mm (coarse-to-fine multi-resolution)
%   Tolerances         : [0.02 0.02 0.02 0.001 0.001 0.001
%                         0.01  0.01  0.01 0.001 0.001 0.001]
%                        (translations, rotations, zooms, shears)
%   Smoothing (FWHM)   : [7 7] mm (source and reference smoothing)
%
%   Reference (target) : T1w anatomical image
%   Source             : Mean realigned-unwarped functional (meanuasub-*)
%   Other images       : Full time series (uasub-*)
%
% EXPECTED INPUTS
% ---------------
%   T1w images: <folder>/ses-<ses>/anat/sub-*_ses-*_T1w.nii
%   Mean functional images: <folder>/ses-<ses>/func/<task>/meanuasub-*.nii
%   Functional time series: <folder>/ses-<ses>/func/<task>/uasub-*.nii
%   fmri2T1wmap: index vector from exclude_orphan_fMRIs.m
%
% OUTPUTS
% -------
%   Coregistration updates the NIfTI headers of the source and other images
%   in-place (no new files are written; only the affine matrix in the
%   header is modified).
%
% REFERENCES
% ----------
%   Collignon A, et al. Automated multi-modality image registration based
%   on information theory. IPMI. 1995;263-274.
%
%   Studholme C, Hill DLG, Hawkes DJ. An overlap invariant entropy measure
%   of 3D medical image alignment. Pattern Recognit. 1999;32(1):71-86.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, fmri2T1wmap, account, burst, mem, time, user,
%   matlabver, slurmfolder
%
% =========================================================================

% Specify the images to be used as targets.
filedir_ref = dir(strrep([folder 'ses-' ses '/anat/sub-*_ses-' ses '*_T1w.nii'],'**','*'));
% Specify the images to be used as sources.
filedir_src = dir(strrep([folder 'ses-' ses '/func/' task '/meanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));
% Specify the images to be coregistered.
filedir_oth = dir(strrep([folder 'ses-' ses '/func/' task '/uasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Rearrange the structural MRIs to match the functional MRIs
filedir_ref = filedir_ref(fmri2T1wmap);

% Create the batches and store them in a cell.
N = numel(filedir_ref);
matlabbatch_all = cell(N,1);
for i = 1:N
    ref = fullfile(filedir_ref(i).folder,filedir_ref(i).name);
    src = fullfile(filedir_src(i).folder,filedir_src(i).name);
    oth = fullfile(filedir_oth(i).folder,filedir_oth(i).name);
    matlabbatch_all{i}.spm.spatial.coreg.estimate.ref = {ref};
    matlabbatch_all{i}.spm.spatial.coreg.estimate.source = {src};
    matlabbatch_all{i}.spm.spatial.coreg.estimate.other = {src,oth}';
    matlabbatch_all{i}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch_all{i}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch_all{i}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch_all{i}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
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