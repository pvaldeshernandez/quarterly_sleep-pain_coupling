function create_lh_atlas_roi(atlas_file, output_dir)
% CREATE_LH_ATLAS_ROI  Create a probabilistic bilateral lateral
%   hypothalamus (LH) ROI from the Neudorfer et al. hypothalamic atlas.
%
% Usage:
%   create_lh_atlas_roi(atlas_file)
%   create_lh_atlas_roi(atlas_file, output_dir)
%
% Inputs:
%   atlas_file  - Path to the Neudorfer hypothalamic atlas label NIfTI
%                 file (e.g., 'hypothalamic_atlas_labels.nii.gz').
%   output_dir  - (Optional) Directory for the output ROI. Default: pwd.
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The lateral hypothalamus (LH) is a key component of the ascending
% arousal system. Orexin/hypocretin neurons in the LH project widely
% to cortex, thalamus, and brainstem, promoting and stabilizing
% wakefulness. Loss of orexin neurons causes narcolepsy.
%
% In the context of pain-sleep coupling, the LH sits downstream of
% the PBN-CeA pain-arousal pathway:
%
%   Nociception -> PBN -> CeA -> LH (orexin neurons) -> wakefulness
%
% Lynch et al. (2025, Advanced Science) proposed that pain-induced
% activation of orexin neurons in the LH is a key mechanism by which
% acute pain disrupts sleep. This makes the LH a candidate moderator
% of pain-to-sleep (PS) coupling: individuals with greater LH gray
% matter volume or stronger LH BOLD responses may show stronger
% pain-to-sleep coupling.
%
% =========================================================================
% ATLAS SOURCE
% =========================================================================
%   Neudorfer C, Germann J, Elias GJB, Gramer R, Boutet A,
%   Lozano AM. A high-resolution in vivo magnetic resonance imaging
%   atlas of the human hypothalamic region. Scientific Data. 2020;
%   7:305. doi:10.1038/s41597-020-00644-6
%
% The Neudorfer atlas provides labeled parcellations of the human
% hypothalamus derived from ultra-high-resolution (0.7 mm) 7T MRI
% data, registered to MNI152 space. The atlas includes discrete labels
% for individual hypothalamic nuclei and subregions.
%
% For the lateral hypothalamus, we use:
%   Label 25: Left lateral hypothalamic area (LHA)
%   Label 26: Right lateral hypothalamic area (LHA)
%
% These labels encompass the region containing orexin/hypocretin
% neurons, though the atlas does not distinguish orexin-producing
% neurons from other LH cell populations.
%
% The atlas is freely available from:
%   https://zenodo.org/record/3942115
%
% =========================================================================
% PROCESSING STEPS
% =========================================================================
%   1. Load the Neudorfer hypothalamic atlas label volume.
%   2. Extract voxels with label 25 (left LH) or 26 (right LH).
%   3. Create a bilateral binary mask (1 = LH, 0 = background).
%   4. Save as a NIfTI file in the atlas's native space.
%
% Like the PBN from the Brainstem Navigator, this is a label-based
% (binary) atlas rather than a continuous probability map. The output
% is therefore binary, not probabilistic. However, the extraction
% function (extract_atlas_roi_values.m) handles binary masks as a
% special case of probability-weighted extraction (all weights = 1).
%
% =========================================================================
% OUTPUT
% =========================================================================
% Filename: roi_atlas_LH_bilateral.nii
% Space:    MNI152, native atlas resolution
% Values:   Binary (0 or 1)
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
    error('create_lh_atlas_roi:noInput', ...
        'Must provide path to Neudorfer hypothalamic atlas NIfTI.');
end
if ~exist(atlas_file, 'file')
    error('create_lh_atlas_roi:fileNotFound', ...
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
LEFT_LH_LABEL  = 25;   % Left lateral hypothalamic area
RIGHT_LH_LABEL = 26;   % Right lateral hypothalamic area

% -------------------------------------------------------------------------
% Load the atlas
% -------------------------------------------------------------------------
fprintf('Loading Neudorfer hypothalamic atlas: %s\n', atlas_file);
V_atlas = spm_vol(atlas_file);
Y_atlas = spm_read_vols(V_atlas);

% -------------------------------------------------------------------------
% Extract LH labels and create bilateral binary mask
% -------------------------------------------------------------------------
mask = double(Y_atlas == LEFT_LH_LABEL | Y_atlas == RIGHT_LH_LABEL);

n_left  = sum(Y_atlas(:) == LEFT_LH_LABEL);
n_right = sum(Y_atlas(:) == RIGHT_LH_LABEL);
n_total = sum(mask(:));

fprintf('  Left LH  (label %d): %d voxels at %.1f mm\n', ...
    LEFT_LH_LABEL, n_left, abs(V_atlas.mat(1,1)));
fprintf('  Right LH (label %d): %d voxels at %.1f mm\n', ...
    RIGHT_LH_LABEL, n_right, abs(V_atlas.mat(1,1)));
fprintf('  Bilateral total:     %d voxels\n', n_total);

% -------------------------------------------------------------------------
% Write the ROI mask
% -------------------------------------------------------------------------
V_out         = V_atlas;
V_out.fname   = fullfile(output_dir, 'roi_atlas_LH_bilateral.nii');
V_out.dt      = [spm_type('uint8') 0];
V_out.pinfo   = [1; 0; 0];
V_out.descrip = sprintf('Bilateral lateral hypothalamus from Neudorfer et al. 2020 (labels %d+%d)', ...
    LEFT_LH_LABEL, RIGHT_LH_LABEL);

if isfield(V_out, 'private')
    V_out = rmfield(V_out, 'private');
end

V_out = spm_create_vol(V_out);
spm_write_vol(V_out, mask);

fprintf('LH ROI written: %s\n', V_out.fname);
fprintf('Reference: Neudorfer et al. Scientific Data 2020;7:305\n');
fprintf('Motivation: Lynch et al. Advanced Science 2025 (LH orexin neurons in pain-arousal)\n');

end
