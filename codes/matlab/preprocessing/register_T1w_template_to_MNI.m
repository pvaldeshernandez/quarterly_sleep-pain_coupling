% REGISTER_T1W_TEMPLATE_TO_MNI  Bring DARTEL template into MNI space and
%                                create tissue maps at multiple resolutions.
%
% =========================================================================
% PIPELINE POSITION: Step 9 (Group C — normalization)
% =========================================================================
%
% PURPOSE
% -------
% The DARTEL template lives in its own coordinate system, which is aligned
% to, but not identical with, MNI space. This script applies the affine
% transformation estimated during deform_all.m (stored in *_2mni.mat) to
% bring the final DARTEL template into standard MNI coordinates, then
% resamples the tissue-probability maps (GM, WM, CSF) at three spatial
% resolutions:
%
%   - 3 mm isotropic (matching the fMRI voxel size)
%   - 2 mm isotropic (matching MNI152 standard resolution)
%   - 1 mm isotropic (high-resolution anatomical analyses)
%
% Additionally, binarized GM masks are created at each resolution using a
% probability threshold of 0.25, for use as explicit masks in group-level
% fMRI and VBM analyses.
%
% PROCESSING STEPS
% -----------------
% 1. Copy final template (_6.nii) to root folder and split into
%    separate c1/c2/c3 volumes.
% 2. Copy SPM's avg152T1.nii (2mm MNI reference) and create a 1mm
%    version via reslice_spacen() and cropImage().
% 3. Load the DARTEL-to-MNI affine from *_2mni.mat.
% 4. Apply the affine (spm_get_space) and resample tissue maps to
%    3mm, 2mm, and 1mm reference spaces using change_spacen().
% 5. Binarize GM maps at threshold=0.25.
%
% EXPECTED INPUTS
% ---------------
%   Final DARTEL template:
%     <folder>/ses-01/anat/<study>_T1w__6.nii
%   Affine transformation:
%     <folder>/ses-01/anat/<study>_T1w__6_2mni.mat
%   A deformed fMRI (for 3mm space definition):
%     <folder>/ses-<ses>/func/<task>/dmeanuasub-*_bold.nii
%
% OUTPUTS
% -------
%   Tissue probability maps in MNI space:
%     <folder>/<study>_T1w__gm-3mm.nii, _wm-3mm.nii, _csf-3mm.nii
%     <folder>/<study>_T1w__gm-2mm.nii, _wm-2mm.nii, _csf-2mm.nii
%     <folder>/<study>_T1w__gm-1mm.nii, _wm-1mm.nii, _csf-1mm.nii
%   Binarized GM masks (threshold=0.25):
%     <folder>/<study>_T1w__gm-3mm-binarized(0.25).nii
%     <folder>/<study>_T1w__gm-2mm-binarized(0.25).nii
%     <folder>/<study>_T1w__gm-1mm-binarized(0.25).nii
%
% DEPENDENCIES
% ------------
%   reslice_spacen.m, cropImage.m (matlab/lib/)
%   change_spacen.m, filedir2cell.m (on MATLAB path)
%
% REFERENCES
% ----------
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, ses, task, study
%
% =========================================================================

% Copy DARTEL template to root folder.
template = fullfile(folder,'ses-01/anat',[study '_T1w__6.nii']);
copyfile(template,folder);
[pp,nn,ee] = fileparts(template);
template = fullfile(folder,[nn ee]);

% Create temporal files to change space
V = spm_vol(template);
P = spm_read_vols(V);
%---
V1 = V(1);
c1 = fullfile(folder,['c1' nn ee]);
V1.fname = c1;
spm_write_vol(V1,P(:,:,:,1));
%---
V2 = V(1);
c2 = fullfile(folder,['c2' nn ee]);
V2.fname = c2;
spm_write_vol(V2,P(:,:,:,2));
%---
V3 = V(1);
c3 = fullfile(folder,['c3' nn ee]);
V3.fname = c3;
spm_write_vol(V3,P(:,:,:,3));

% Copy reference images in MNI 2mm resolution
MNI2 = fullfile(fileparts(which('spm')),'canonical','avg152T1.nii');
copyfile(MNI2,folder);
MNI2 = fullfile(folder,'avg152T1.nii');
% Create reference images in MNI 1mm resolution
MNI1 = fullfile(folder,'ravg152T1.nii');
reslice_spacen(MNI2,[1 1 1],1);
cropImage(MNI1,[-1 -1 -1],0,0);

% Load the affine transformation from the DARTEL to the MNI space.
aff_file = fullfile(folder,'ses-01/anat',[study '_T1w__6_2mni.mat']);
load(aff_file);

% Load an fMRI to define 3mm space
dfMRIs = filedir2cell(dir(strrep([folder 'ses-' ses '/func/' task '/dmeanuasub-*_ses-' ses '*_task-' task '*_bold.nii'],'**','*')));
fMRI3 = dfMRIs{1};

% Create DARTEL templates in MNI space
c1template3 = fullfile(folder,[study '_T1w__gm-3mm.nii']);
c2template3 = fullfile(folder,[study '_T1w__wm-3mm.nii']);
c3template3 = fullfile(folder,[study '_T1w__csf-3mm.nii']);
c1template2 = fullfile(folder,[study '_T1w__gm-2mm.nii']);
c2template2 = fullfile(folder,[study '_T1w__wm-2mm.nii']);
c3template2 = fullfile(folder,[study '_T1w__csf-2mm.nii']);
c1template1 = fullfile(folder,[study '_T1w__gm-1mm.nii']);
c2template1 = fullfile(folder,[study '_T1w__wm-1mm.nii']);
c3template1 = fullfile(folder,[study '_T1w__csf-1mm.nii']);
for i = 1:3
    % Change space of template from the DARTEL to the MNI space.
    warning off
    spm_get_space(c1,mni.affine);
    spm_get_space(c2,mni.affine);
    spm_get_space(c3,mni.affine);
    warning on
    % Resample 3 mm
    tmpc13 = change_spacen(c1,fMRI3,1);
    tmpc23 = change_spacen(c2,fMRI3,1);
    tmpc33 = change_spacen(c3,fMRI3,1);
    % Rename 1 mm resampled template
    movefile(tmpc13,c1template3);
    movefile(tmpc23,c2template3);
    movefile(tmpc33,c3template3);
    % Resample 2 mm
    tmpc12 = change_spacen(c1,MNI2,1);
    tmpc22 = change_spacen(c2,MNI2,1);
    tmpc32 = change_spacen(c3,MNI2,1);
    % Rename 1 mm resampled template
    movefile(tmpc12,c1template2);
    movefile(tmpc22,c2template2);
    movefile(tmpc32,c3template2);
    % Resample to 1 mm
    tmpc11 = change_spacen(c1,MNI1,1);
    tmpc21 = change_spacen(c2,MNI1,1);
    tmpc31 = change_spacen(c3,MNI1,1);
    % Rename 1 mm resampled template
    movefile(tmpc11,c1template1);
    movefile(tmpc21,c2template1);
    movefile(tmpc31,c3template1);
end
delete(c1)
delete(c2)
delete(c3)
delete(MNI2)
delete(MNI1)
delete(template);

% Binarize GM masks.
th = .25;
% In 3-mm resampled template space
V = spm_vol(c1template3);
P = spm_read_vols(V);
P = P>th;
V.fname = fullfile(folder,[study sprintf('_T1w__gm-3mm-binarized(%.2f).nii',th)]);
spm_write_vol(V,P);
% In 2-mm resampled template space
V = spm_vol(c1template2);
P = spm_read_vols(V);
P = P>th;
V.fname = fullfile(folder,[study sprintf('_T1w__gm-2mm-binarized(%.2f).nii',th)]);
spm_write_vol(V,P);
% In 1-mm resampled template space
V = spm_vol(c1template1);
P = spm_read_vols(V);
P = P>th;
V.fname = fullfile(folder,[study sprintf('_T1w__gm-1mm-binarized(%.2f).nii',th)]);
spm_write_vol(V,P);