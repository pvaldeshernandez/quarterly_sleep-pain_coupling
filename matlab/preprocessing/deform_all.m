% DEFORM_ALL  Apply DARTEL deformations to all structural and functional
%             images via push-forward warping.
%
% =========================================================================
% PIPELINE POSITION: Step 8 (Group C — normalization)
% =========================================================================
%
% PURPOSE
% -------
% Warp all native-space images into the study-specific DARTEL template
% space using the flow fields estimated in create_T1w_template.m and
% get_deformations.m. This script uses SPM's "Deformations" utility with
% DARTEL flow fields and push-forward warping, which maps each native
% voxel to its template-space location.
%
% Three sets of deformations are applied:
%
% (A) T1w stream (indirect, via T1w flow fields):
%     - Unsmoothed: T1w + tissue maps c1-c5 at 1mm (prefix 'd')
%     - Smoothed VBM: c1 + c2 with Jacobian modulation and 8mm FWHM
%       smoothing (prefix 'smd') for voxel-based morphometry
%
% (B) fMRI stream — indirect (via T1w flow fields, mapped through
%     fmri2T1wmap):
%     - Unsmoothed: mean fMRI + time series at 3mm isotropic (prefix 'd')
%     - Smoothed: time series at 3mm with 6mm FWHM smoothing (prefix 'sd')
%       for connectivity/activation analyses
%
% (C) fMRI stream — direct (via fMRI-derived flow fields):
%     - Same as (B) but using flow fields from the fMRI segmentations
%       (prefixes 'd2', 'sd2'). Provides an alternative normalization
%       pathway for quality comparison.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Warping method       : Push-forward (native -> template)
%   Flow field K         : 6 (2^6 = 64 time steps for diffeomorphic
%                          integration; matches final DARTEL iteration)
%   Template             : Final DARTEL template (<study>_T1w__6.nii)
%
%   T1w deformations:
%     Voxel size (unsmoothed)  : [1 1 1] mm
%     Preserve                 : 0 (concentrations; no Jacobian modulation)
%     Smoothing (unsmoothed)   : [0 0 0] mm
%     Preserve (VBM)           : 1 (Jacobian modulation for volume analysis)
%     Smoothing (VBM)          : [8 8 8] mm FWHM
%
%   fMRI deformations:
%     Voxel size               : [3 3 3] mm isotropic
%     Preserve                 : 0 (no Jacobian modulation)
%     Smoothing (unsmoothed)   : [0 0 0] mm
%     Smoothing (smoothed)     : [6 6 6] mm FWHM
%
% EXPECTED INPUTS
% ---------------
%   T1w flow fields: <folder>/ses-<ses>/anat/u_rc1sub-*_T1w.nii
%   T1w images: sub-*_T1w.nii, c1-c5 tissue maps
%   fMRI flow fields (indirect): same as T1w, indexed via fmri2T1wmap
%   fMRI flow fields (direct): <folder>/ses-<ses>/func/<task>/u_rc1meanuasub-*
%   fMRI images: uasub-* (time series), meanuasub-* (mean)
%   Final template: <folder>/ses-01/anat/<study>_T1w__6.nii
%
% OUTPUTS
% -------
%   Normalized images in template space:
%     d*.nii    — Unsmoothed T1w/fMRI (indirect)
%     smd*.nii  — Smoothed modulated VBM tissue maps
%     sd*.nii   — Smoothed fMRI (indirect, 6mm)
%     d2*.nii   — Unsmoothed fMRI (direct)
%     sd2*.nii  — Smoothed fMRI (direct, 6mm)
%
% REFERENCES
% ----------
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, study, fmri2T1wmap, account, burst, mem, time,
%   user, matlabver, slurmfolder
%
% =========================================================================

% Specify images to be deformed.
filedir_T1ws = dir(strrep([folder 'ses-' ses '/anat/sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_c1s = dir(strrep([folder 'ses-' ses '/anat/c1sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_c2s = dir(strrep([folder 'ses-' ses '/anat/c2sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_c3s = dir(strrep([folder 'ses-' ses '/anat/c3sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_c4s = dir(strrep([folder 'ses-' ses '/anat/c4sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_c5s = dir(strrep([folder 'ses-' ses '/anat/c5sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_meanfMRIs = dir(strrep([folder 'ses-' ses '/func/' task '/meanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));
filedir_fMRIs = dir(strrep([folder 'ses-' ses '/func/' task '/uasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Specify template.
template = fullfile(folder,'ses-01/anat',[study '_T1w__6.nii']);

% Specify flowfieds from T1ws
filedir_flows_T1ws = dir(strrep([folder 'ses-' ses '/anat/u_rc1sub-*_ses-' ses '*_T1w.nii'],'**','*'));

% Create the batches and store them in a cell.
N1 = numel(filedir_flows_T1ws);
matlabbatch_T1w = cell(N1,1);
for i = 1:N1
    % Configure deformations for T1ws.
    matlabbatch_T1w{i}.spm.util.defs.comp{1}.dartel.flowfield = {fullfile(filedir_flows_T1ws(i).folder,filedir_flows_T1ws(i).name)};
    matlabbatch_T1w{i}.spm.util.defs.comp{1}.dartel.times = [0 1];
    matlabbatch_T1w{i}.spm.util.defs.comp{1}.dartel.K = 6;
    matlabbatch_T1w{i}.spm.util.defs.comp{1}.dartel.template = {template};
    % Specify T1ws to warp unsmoothed.
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.fnames = {
                                                             fullfile(filedir_T1ws(i).folder,filedir_T1ws(i).name)
                                                             fullfile(filedir_c1s(i).folder,filedir_c1s(i).name)
                                                             fullfile(filedir_c2s(i).folder,filedir_c2s(i).name)
                                                             fullfile(filedir_c3s(i).folder,filedir_c3s(i).name)
                                                             fullfile(filedir_c4s(i).folder,filedir_c4s(i).name)
                                                             fullfile(filedir_c5s(i).folder,filedir_c5s(i).name)
                                                             };
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.weight = {''};
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.savedir.savesrc = 1;
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.fov.bbvox.vox = [1 1 1];
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.preserve = 0;
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.fwhm = [0 0 0];
    matlabbatch_T1w{i}.spm.util.defs.out{1}.push.prefix = 'd';
    % Specify T1ws to warp smoothed for VBM.
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.fnames = {
                                                             fullfile(filedir_c1s(i).folder,filedir_c1s(i).name)
                                                             fullfile(filedir_c2s(i).folder,filedir_c2s(i).name)
                                                             };
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.weight = {''};
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.savedir.savesrc = 1;
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.fov.bbvox.vox = [1 1 1];
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.preserve = 1;
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.fwhm = [8 8 8];
    matlabbatch_T1w{i}.spm.util.defs.out{2}.push.prefix = 'smd';
end

% Specify flowfieds from T1ws for indirect fMRIs
filedir_flows_fMRI = filedir_flows_T1ws(fmri2T1wmap);

% Create the batches and store them in a cell.
N2 = numel(filedir_flows_fMRI);
matlabbatch_fMRI = cell(N2,1);
for i = 1:N2
    % Configure deformations for fMRIs.
    matlabbatch_fMRI{i}.spm.util.defs.comp{1}.dartel.flowfield = {fullfile(filedir_flows_fMRI(i).folder,filedir_flows_fMRI(i).name)};
    matlabbatch_fMRI{i}.spm.util.defs.comp{1}.dartel.times = [0 1];
    matlabbatch_fMRI{i}.spm.util.defs.comp{1}.dartel.K = 6;
    matlabbatch_fMRI{i}.spm.util.defs.comp{1}.dartel.template = {template};
    % Specify fMRIs to warp unsmoothed.
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.fnames = {
                                                        fullfile(filedir_fMRIs(i).folder,filedir_fMRIs(i).name)
                                                        fullfile(filedir_meanfMRIs(i).folder,filedir_meanfMRIs(i).name)
                                                        };
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.weight = {''};
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.savedir.savesrc = 1;
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.fov.bbvox.vox = [3 3 3];
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.preserve = 0;
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.fwhm = [0 0 0];
    matlabbatch_fMRI{i}.spm.util.defs.out{1}.push.prefix = 'd';
    % Specify fMRIs to warp smoothed for connectivity.
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.fnames = {fullfile(filedir_fMRIs(i).folder,filedir_fMRIs(i).name)};
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.weight = {''};
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.savedir.savesrc = 1;
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.fov.bbvox.vox = [3 3 3];
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.preserve = 0;
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.fwhm = [6 6 6];
    matlabbatch_fMRI{i}.spm.util.defs.out{2}.push.prefix = 'sd';
end

% Specify flowfieds for direct fMRIs
filedir_flows_fMRI = dir(strrep([folder 'ses-' ses '/func/' task '/u_rc1meanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

N3 = numel(filedir_flows_fMRI);
matlabbatch_fMRI2 = cell(N3,1);
for i = 1:N3
    % Configure deformations for fMRIs.
    matlabbatch_fMRI2{i}.spm.util.defs.comp{1}.dartel.flowfield = {fullfile(filedir_flows_fMRI(i).folder,filedir_flows_fMRI(i).name)};
    matlabbatch_fMRI2{i}.spm.util.defs.comp{1}.dartel.times = [0 1];
    matlabbatch_fMRI2{i}.spm.util.defs.comp{1}.dartel.K = 6;
    matlabbatch_fMRI2{i}.spm.util.defs.comp{1}.dartel.template = {template};
    % Specify fMRIs to warp unsmoothed.
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.fnames = {
                                                            fullfile(filedir_meanfMRIs(i).folder,filedir_meanfMRIs(i).name)
                                                            fullfile(filedir_fMRIs(i).folder,filedir_fMRIs(i).name)
                                                            };
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.weight = {''};
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.savedir.savesrc = 1;
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.fov.bbvox.vox = [3 3 3];
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.preserve = 0;
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.fwhm = [0 0 0];
    matlabbatch_fMRI2{i}.spm.util.defs.out{1}.push.prefix = 'd2';
    % Specify fMRIs to warp smoothed for connectivity.
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.fnames = {fullfile(filedir_fMRIs(i).folder,filedir_fMRIs(i).name)};
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.weight = {''};
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.savedir.savesrc = 1;
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.fov.bbvox.bb = NaN(2,3);
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.fov.bbvox.vox = [3 3 3];
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.preserve = 0;
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.fwhm = [6 6 6];
    matlabbatch_fMRI2{i}.spm.util.defs.out{2}.push.prefix = 'sd2';
end

matlabbatch_all = [matlabbatch_T1w; matlabbatch_fMRI; matlabbatch_fMRI2];

copydeformedjsons(filedir_fMRIs,{'d','sd','d2','sd2'})
copydeformedjsons(filedir_T1ws,{'d'})

% Run the first batch to create the Affine transformation from the DARTEL
% to the MNI space
matlabbatch = matlabbatch_all(1);
spm_jobman('run',matlabbatch(1));

% Initialize the cells containing the files needed to run the batches in
% the SLURM.
N = numel(matlabbatch_all);
matoutfiles = cell(N,1);
shfiles = cell(N,1);
logfiles = cell(N,1);
matinfiles = cell(N,1);
strrun = 'run';
conf = struct('account',account,'burst',burst,'mem',mem,'time',time,'user',user,'matlabver',matlabver);
% Send the rest of batches to SLURM.
for i = 2:N
    matlabbatch = matlabbatch_all(i);
    [matoutfiles{i},shfiles{i},logfiles{i},matinfiles{i}] = sendslurm('','spm_jobman',slurmfolder,conf,strrun,matlabbatch);
end
% Clean the files needed to run the batches in the SLURM.
chckdelslurm(matoutfiles(2:N),shfiles(2:N),logfiles(2:N),matinfiles(2:N),slurmfolder,1);