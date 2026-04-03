% EXCLUDE_ORPHAN_FMRIS  Remove functional scans that lack a matching T1w.
%
% =========================================================================
% PIPELINE POSITION: Step 0 (run before all other steps)
% =========================================================================
%
% PURPOSE
% -------
% In a multi-session BIDS dataset, some sessions may have functional scans
% but no corresponding T1-weighted anatomical image (e.g., due to scan
% abortion or QC exclusion). Because every downstream step (coregistration,
% indirect DARTEL normalization) requires a subject-matched T1w, these
% "orphan" functional files must be identified and removed before the
% pipeline proceeds.
%
% WHAT IT DOES
% ------------
% 1. Lists all T1w and BOLD NIfTI files matching the session/task filters.
% 2. Extracts subject-session identifiers (sub-XXX_ses-YY) from filenames.
% 3. Matches each functional scan to a T1w using MATLAB's ismember().
% 4. Moves unmatched functional scans into a parallel '-excluded' directory
%    so they are preserved but not processed.
% 5. Produces the variable 'fmri2T1wmap', an index vector that maps each
%    retained functional scan to its corresponding T1w. This variable is
%    used by coregister_fMIRs_to_T1ws.m and deform_all.m.
%
% EXPECTED INPUTS
% ---------------
%   - T1w images: <folder>/ses-<ses>/anat/sub-*_ses-*_T1w.nii
%   - BOLD images: <folder>/ses-<ses>/func/<task>/sub-*_ses-*_task-*_bold.nii
%
% OUTPUTS
% -------
%   - fmri2T1wmap : [Nf x 1] vector of indices into the T1w file list
%   - Orphan BOLD files moved to <folder>-excluded/ses-<ses>/func/<task>/
%
% DEPENDENCIES
% ------------
%   - condmkdir.m (matlab/lib/)
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task
%
% =========================================================================

% List the queried structural and functional MRIs.
filedir_T1ws = dir(strrep([folder 'ses-' ses '/anat/sub-*_ses-' ses '*_T1w.nii'],'**','*'));
filedir_fMRIs = dir(strrep([folder 'ses-' ses '/func/' task '/sub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*'));

% Obtain subject and session information.
Ns = numel(filedir_T1ws);
names_T1w = cell(Ns,1);
for i = 1:Ns
    tmp = split(filedir_T1ws(i).name,'_');
    tmp = tmp(1:2);
    names_T1w{i} = [tmp{1} '_' tmp{2}];
end
Nf = numel(filedir_fMRIs);
names_fMRI = cell(Nf,1);
for i = 1:Nf
    tmp = split(filedir_fMRIs(i).name,'_');
    tmp = tmp(1:2);
    names_fMRI{i} = [tmp{1} '_' tmp{2}];
end

% Match queried functional MRIs with structural MRIs
[ind,fmri2T1wmap] = ismember(names_fMRI,names_T1w);

% Exclude the functional MRIs that don't have a structural MRI
for i = 1:numel(ind)
    if ~ind(i)
        current_folder = filedir_fMRIs(i).folder;
        exclude_folder = strrep(current_folder,folder(1:end-1),[folder(1:end-1) '-excluded']);
        name = filedir_fMRIs(i).name;
        condmkdir(exclude_folder);
        movefile(fullfile(current_folder,name),fullfile(exclude_folder,name));
    end
end
fmri2T1wmap(~ind) = [];