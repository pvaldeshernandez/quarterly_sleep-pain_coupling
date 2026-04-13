function create_pbn_atlas_roi(atlas_file, output_dir)
% CREATE_PBN_ATLAS_ROI  Create a probabilistic bilateral parabrachial
%   nucleus (PBN) ROI from the Brainstem Navigator atlas.
%
% Usage:
%   create_pbn_atlas_roi(atlas_file)
%   create_pbn_atlas_roi(atlas_file, output_dir)
%
% Inputs:
%   atlas_file  - Path to the Brainstem Navigator label atlas NIfTI file
%                 (e.g., 'BrainstemNavigator_labels_v0.9.nii.gz').
%   output_dir  - (Optional) Directory for the output ROI. Default: pwd.
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The parabrachial nucleus (PBN) is a brainstem structure at the junction
% of the pons and midbrain that serves as a critical relay in ascending
% pain and arousal pathways. Specifically, nociceptive signals from the
% spinal dorsal horn reach the PBN via the spinoparabrachial tract,
% where PBN^elCGRP neurons (expressing calcitonin gene-related peptide)
% project to the central nucleus of the amygdala (CeA), lateral
% hypothalamus, and bed nucleus of the stria terminalis (BNST).
%
% This PBN-to-forebrain pathway was identified as the origin of the
% pain-induced wakefulness circuit in:
%
%   Lynch AC, Ma H, Bhatt RR, et al. Pain-induced wakefulness: A
%   comprehensive framework for the study of pain-sleep interaction.
%   Advanced Science. 2025. doi:10.1002/advs.202415872
%
% Lynch et al. proposed that the PBN^elCGRP pathway drives transitions
% from sleep to wakefulness in response to nociceptive input, which
% forms the physiological basis for testing PBN as a moderator of
% pain-to-sleep (PS) coupling in our model.
%
% =========================================================================
% ATLAS SOURCE
% =========================================================================
% Brainstem Navigator atlas, version 0.9
%
%   Singh K, Indovina I, Augustinack JC, et al. An optimized
%   probabilistic atlas of brainstem nuclei in the human brain using
%   the EagleVAC (Evaluate And Grade Localization Entities by Voting
%   Among Classes) framework. (2024, in preparation)
%
% The atlas provides integer labels for brainstem nuclei in MNI152
% space at 0.5 mm resolution. For the PBN, we use:
%
%   Label 19: Left lateral parabrachial nucleus
%   Label 20: Right lateral parabrachial nucleus
%
% These labels target the lateral subdivision of the PBN, which
% contains the nociceptive relay neurons described by Lynch et al.
%
% The atlas is freely available for academic use from:
%   https://www.nitrc.org/projects/brainstemnavig/
%
% =========================================================================
% PROCESSING STEPS
% =========================================================================
%   1. Load the Brainstem Navigator label volume.
%   2. Extract voxels with label 19 (left PBN) or 20 (right PBN).
%   3. Create a bilateral binary mask (1 = PBN, 0 = background).
%   4. Save as a NIfTI file in the atlas's native space (0.5 mm MNI152).
%
% The output mask is BINARY (not probabilistic) because the Brainstem
% Navigator provides discrete labels rather than probability maps. The
% term "probabilistic" in the atlas refers to the multi-observer voting
% framework used to derive the labels; the final label image is a hard
% assignment.
%
% At 3 mm fMRI resolution, this bilateral PBN mask covers approximately
% 3 voxels, reflecting the small physical size of the nucleus (~2 mm
% diameter per side). This small footprint limits statistical power but
% is anatomically accurate.
%
% =========================================================================
% OUTPUT
% =========================================================================
% Filename: roi_atlas_PBN_bilateral.nii
% Space:    MNI152, 0.5 mm isotropic (native atlas resolution)
% Values:   Binary (0 or 1)
%
% Note: For extraction at fMRI resolution, the calling code must
% resample this mask to the target functional image grid or use
% nearest-neighbor interpolation.
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (for spm_vol, spm_read_vols, spm_create_vol, spm_write_vol)
%
% =========================================================================
% Author: Pedro Valdes-Hernandez, University of Florida, 2025-2026
% =========================================================================

% -------------------------------------------------------------------------
% Input validation
% -------------------------------------------------------------------------
if nargin < 1 || isempty(atlas_file)
    error('create_pbn_atlas_roi:noInput', ...
        'Must provide path to Brainstem Navigator label atlas NIfTI.');
end
if ~exist(atlas_file, 'file')
    error('create_pbn_atlas_roi:fileNotFound', ...
        'Atlas file not found: %s', atlas_file);
end
if nargin < 2 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% Label definitions
% -------------------------------------------------------------------------
LEFT_PBN_LABEL  = 19;   % Left lateral parabrachial nucleus
RIGHT_PBN_LABEL = 20;   % Right lateral parabrachial nucleus

% -------------------------------------------------------------------------
% Load the atlas
% -------------------------------------------------------------------------
fprintf('Loading Brainstem Navigator atlas: %s\n', atlas_file);
V_atlas = spm_vol(atlas_file);
Y_atlas = spm_read_vols(V_atlas);

% -------------------------------------------------------------------------
% Extract PBN labels and create bilateral binary mask
% -------------------------------------------------------------------------
mask = double(Y_atlas == LEFT_PBN_LABEL | Y_atlas == RIGHT_PBN_LABEL);

n_left  = sum(Y_atlas(:) == LEFT_PBN_LABEL);
n_right = sum(Y_atlas(:) == RIGHT_PBN_LABEL);
n_total = sum(mask(:));

fprintf('  Left PBN  (label %d): %d voxels at %.1f mm\n', ...
    LEFT_PBN_LABEL, n_left, abs(V_atlas.mat(1,1)));
fprintf('  Right PBN (label %d): %d voxels at %.1f mm\n', ...
    RIGHT_PBN_LABEL, n_right, abs(V_atlas.mat(1,1)));
fprintf('  Bilateral total:      %d voxels\n', n_total);

% -------------------------------------------------------------------------
% Write the ROI mask
% -------------------------------------------------------------------------
V_out         = V_atlas;
V_out.fname   = fullfile(output_dir, 'roi_atlas_PBN_bilateral.nii');
V_out.dt      = [spm_type('uint8') 0];
V_out.pinfo   = [1; 0; 0];
V_out.descrip = sprintf('Bilateral lateral PBN from Brainstem Navigator (labels %d+%d)', ...
    LEFT_PBN_LABEL, RIGHT_PBN_LABEL);

% Remove SPM private field if present (prevents write errors)
if isfield(V_out, 'private')
    V_out = rmfield(V_out, 'private');
end

V_out = spm_create_vol(V_out);
spm_write_vol(V_out, mask);

fprintf('PBN ROI written: %s\n', V_out.fname);
fprintf('Reference: Singh et al. (Brainstem Navigator / EagleVAC)\n');
fprintf('Motivation: Lynch et al. Advanced Science 2025 (PBN^elCGRP pain-arousal pathway)\n');

end
