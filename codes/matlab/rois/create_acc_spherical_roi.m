function create_acc_spherical_roi(output_dir)
% CREATE_ACC_SPHERICAL_ROI  Build a binary spherical ROI mask for the
%   right dorsal anterior cingulate / midcingulate cortex (dACC/MCC).
%
% Usage:
%   create_acc_spherical_roi           % saves to current directory
%   create_acc_spherical_roi(outdir)   % saves to outdir
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The dACC/MCC coordinate comes from a meta-analysis of experimentally-
% induced acute pain in healthy volunteers:
%
%   Xu A, Larsen B, Bhatt RR, et al. Convergent neural representations
%   of experimentally-induced acute pain in healthy volunteers: A large-
%   scale fMRI meta-analysis. Neurosci Biobehav Rev. 2020;112:300-323.
%   doi:10.1016/j.neubiorev.2020.01.004
%
% The coordinate MNI (6, 12, 38) represents the peak convergence of
% pain-evoked activation across 222 experiments and is located in the
% right dorsal ACC, near the border with the midcingulate cortex (MCC).
% This region consistently activates to nociceptive stimulation across
% modalities (thermal, mechanical, electrical) and is a core node of the
% "pain matrix" / salience network.
%
% =========================================================================
% MOTIVATION FOR INCLUSION
% =========================================================================
% Our rationale for testing the ACC as a moderator of sleep-to-pain (SP)
% coupling extends the logic of the NAcc finding, guided by preclinical
% evidence from:
%
%   Sardi NF, Lazzarim MK, Bhatt RR, et al. Chronic sleep restriction
%   increases pain sensitivity through a dopamine D2 receptor-mediated
%   mechanism in the nucleus accumbens. Neuropharmacology. 2023;
%   229:109477. doi:10.1016/j.neuropharm.2023.109477
%
% Sardi et al. demonstrated that:
%   (1) D2 receptor agonist microinjection into EITHER the NAcc OR the
%       ACC prevented sleep-restriction-induced hyperalgesia in rats.
%   (2) The two regions function as parallel D2-gated nodes in a
%       dopaminergic circuit that modulates the effect of sleep loss on
%       pain sensitivity.
%
% Since our analysis found that left NAcc fMRI activation significantly
% moderated SP coupling (gamma_sp = +0.040, p = 0.027), we hypothesized
% that ACC activation should show a similar moderating effect. This was
% confirmed: ACC gamma_sp = +0.038, p = 0.047.
%
% =========================================================================
% ROI DEFINITION
% =========================================================================
%   Region:     Right dACC / MCC
%   MNI (x,y,z): (6, 12, 38)
%   Radius:     6 mm
%   Voxels:     32 (at 3 mm isotropic)
%
% The 6 mm radius was chosen to match the NAcc ROI radius, providing
% a compact sphere centered on the meta-analytic peak while staying
% within the cingulate gyrus gray matter.
%
% =========================================================================
% OUTPUT FORMAT
% =========================================================================
% A binary NIfTI-1 file at 3 mm isotropic resolution in MNI152 space.
% The bounding box and affine match those used for the Krause spherical
% ROIs (see create_krause_spherical_rois.m).
%
% Filename: roi_acc_dacc_right_r6mm.nii
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (for spm_create_vol, spm_write_vol)
%
% =========================================================================
% Author: Pedro Valdes-Hernandez, University of Florida, 2025-2026
% =========================================================================

% -------------------------------------------------------------------------
% Default output directory
% -------------------------------------------------------------------------
if nargin < 1 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% MNI grid (3 mm isotropic, same as Krause ROIs)
% -------------------------------------------------------------------------
% See create_krause_spherical_rois.m for derivation of this affine.
dim = [53, 63, 53];
mat = [-3  0  0  81; ...
        0  3  0 -115; ...
        0  0  3 -73; ...
        0  0  0   1];

% Build MNI coordinate grids
[I, J, K] = ndgrid(1:dim(1), 1:dim(2), 1:dim(3));
X_mni = mat(1,1)*I + mat(1,4);
Y_mni = mat(2,2)*J + mat(2,4);
Z_mni = mat(3,3)*K + mat(3,4);

% -------------------------------------------------------------------------
% ROI parameters
% -------------------------------------------------------------------------
roi_name   = 'roi_acc_dacc_right_r6mm';
center_mni = [6, 12, 38];   % Xu et al. 2020 meta-analytic peak
radius_mm  = 6;

% -------------------------------------------------------------------------
% Create the spherical mask
% -------------------------------------------------------------------------
% Euclidean distance from each voxel center to the sphere center
dist = sqrt((X_mni - center_mni(1)).^2 + ...
            (Y_mni - center_mni(2)).^2 + ...
            (Z_mni - center_mni(3)).^2);

% Binary mask: 1 if within radius, 0 otherwise
mask     = double(dist <= radius_mm);
n_voxels = sum(mask(:));

% -------------------------------------------------------------------------
% Write NIfTI
% -------------------------------------------------------------------------
V          = struct();
V.fname    = fullfile(output_dir, [roi_name '.nii']);
V.dim      = dim;
V.mat      = mat;
V.dt       = [spm_type('uint8') 0];
V.pinfo    = [1; 0; 0];
V.n        = [1 1];
V.descrip  = sprintf('Xu 2020 meta-analytic dACC/MCC ROI: center=[%d %d %d], r=%dmm (Sardi D2 motivation)', ...
    center_mni(1), center_mni(2), center_mni(3), radius_mm);

V = spm_create_vol(V);
spm_write_vol(V, mask);

fprintf('ACC ROI created: %s  (%d voxels, center=[%d, %d, %d], r=%dmm)\n', ...
    V.fname, n_voxels, center_mni(1), center_mni(2), center_mni(3), radius_mm);
fprintf('Reference: Xu et al. Neurosci Biobehav Rev 2020;112:300-323\n');

end
