% SLICE_TIMING  Correct for differences in slice acquisition times.
%
% =========================================================================
% PIPELINE POSITION: Step 3 (Group B — functional stream)
% =========================================================================
%
% PURPOSE
% -------
% EPI volumes are acquired one slice at a time over the duration of a
% single TR. Consequently, each slice samples brain activity at a slightly
% different time point. Slice-timing correction interpolates the signal in
% each slice to a common reference time (the middle slice), removing this
% temporal offset before subsequent analyses that assume simultaneous
% acquisition of all slices within a volume.
%
% SPM PARAMETERS (matching manuscript)
% -------------------------------------
%   Slice order    : Philips interleaved acquisition
%                    Packages of sqrt(nslices) interleaved groups, each
%                    stepping by ds = round(sqrt(nslices)). For example,
%                    with 36 slices (ds=6): [1,7,13,19,25,31, 2,8,14,...].
%   Reference slice: Middle slice (round(nslices/2))
%   TR             : Read from NIfTI header (V.private.timing.tspace)
%   TA             : (1 - 1/nslices) * TR
%   Prefix         : 'a' (slice-timing corrected files begin with 'a')
%
% RATIONALE FOR PHILIPS INTERLEAVED ORDER
% ----------------------------------------
% Philips scanners use a package-based interleaved scheme rather than the
% simple odd-even interleaving used by Siemens/GE. The step size ds
% (approximately sqrt(nslices)) is chosen so that adjacent slices are never
% acquired consecutively, minimizing cross-slice excitation artifacts.
% The exact order is reconstructed from the NIfTI header's slice count.
%
% EXPECTED INPUTS
% ---------------
%   Raw BOLD NIfTI files:
%     <folder>/ses-<ses>/func/<task>/sub-*_ses-*_task-*_bold.nii
%
% OUTPUTS
% -------
%   Slice-timing corrected images (prefix 'a'):
%     <folder>/ses-<ses>/func/<task>/asub-*_ses-*_task-*_bold.nii
%
% REFERENCES
% ----------
%   Sladky R, et al. Slice-timing effects and their correction in
%   functional MRI. NeuroImage. 2011;58(2):588-594.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, account, burst, mem, time, user, matlabver,
%   slurmfolder
%
% =========================================================================

% Specify the fMRI files to be slice timed.
filedir = dir(strrep([folder 'ses-' ses '/func/' task '/sub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Create the batches and store them in a cell.
N = numel(filedir);
matlabbatch_all = cell(N,1);
for i = 1:N
    fMRI = fullfile(filedir(i).folder,filedir(i).name);
    V = spm_vol([fMRI ',1']);
    nslices = V.dim(3);
    TR = V.private.timing.tspace;
    % Define Phillips' slice order.
    ds = round(sqrt(nslices));
    sliceorder = cell(1,ds);
    for s = 1:ds
        sliceorder{s} = s:ds:nslices;
    end
    sliceorder = cat(2,sliceorder{:});
    %---
    matlabbatch_all{i}.spm.temporal.st.scans = {{fMRI}};
    matlabbatch_all{i}.spm.temporal.st.nslices = nslices;
    matlabbatch_all{i}.spm.temporal.st.tr = TR;
    matlabbatch_all{i}.spm.temporal.st.ta = (1-1/nslices)*TR;
    matlabbatch_all{i}.spm.temporal.st.so = sliceorder;
    matlabbatch_all{i}.spm.temporal.st.refslice = round(nslices/2);
    matlabbatch_all{i}.spm.temporal.st.prefix = 'a';
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