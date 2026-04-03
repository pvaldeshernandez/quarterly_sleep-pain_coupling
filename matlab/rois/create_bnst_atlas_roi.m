function create_bnst_atlas_roi(atlas_file, output_dir)
% CREATE_BNST_ATLAS_ROI  Create a probabilistic bed nucleus of the stria
%   terminalis (BNST) ROI from the Theiss et al. atlas.
%
% Usage:
%   create_bnst_atlas_roi(atlas_file)
%   create_bnst_atlas_roi(atlas_file, output_dir)
%
% Inputs:
%   atlas_file  - Path to the Theiss BNST probability map NIfTI file.
%   output_dir  - (Optional) Directory for the output ROI. Default: pwd.
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The bed nucleus of the stria terminalis (BNST) is a forebrain
% structure located at the junction of the limbic system and the
% hypothalamus. It is considered part of the "extended amygdala" and
% shares dense reciprocal connections with the central amygdala (CeA),
% lateral hypothalamus, and brainstem arousal nuclei.
%
% The BNST plays a dual role relevant to pain-sleep coupling:
%   (1) It receives nociceptive input from the PBN/CeA pathway and
%       contributes to sustained pain-related affective states
%       (anxiety-like responding to pain).
%   (2) It modulates arousal via projections to the ventral tegmental
%       area and locus coeruleus.
%
% Lynch et al. (2025, Advanced Science) included the BNST as one of
% five key nodes in the ascending pain-induced wakefulness circuit,
% positioned between the CeA and hypothalamic arousal centers.
%
% =========================================================================
% ATLAS SOURCE
% =========================================================================
%   Theiss JD, Ridgewell C, McHugo M, Heckers S, Blackford JU.
%   Manual segmentation of the human bed nucleus of the stria
%   terminalis using 3T MRI. NeuroImage. 2017;146:288-292.
%   doi:10.1016/j.neuroimage.2016.11.047
%
% The Theiss atlas provides a probabilistic map of the BNST derived
% from manual segmentation of 3T MRI data in a group of healthy
% adults, registered to MNI152 space. The probability values (0 to 1)
% represent the proportion of subjects in whom a given voxel was
% labeled as BNST.
%
% The atlas is available from the authors upon request. See also:
%   Avery SN, Clauss JA, Winder DG, Woodward N, Heckers S,
%   Blackford JU. BNST neurocircuitry in humans. NeuroImage. 2014.
%
% =========================================================================
% PROCESSING STEPS
% =========================================================================
%   1. Load the BNST probability map.
%   2. Threshold at p > 0 (retain all nonzero-probability voxels).
%      As with the SI/BF ROI, the probability values will serve as
%      weights during extraction, so a liberal threshold is appropriate.
%   3. Save the probability map as the ROI mask.
%
% The BNST is a small structure (~5-6 mm in longest dimension), so
% even the thresholded map contains relatively few voxels.
%
% =========================================================================
% OUTPUT
% =========================================================================
% Filename: roi_atlas_BNST_bilateral.nii
% Space:    MNI152, native atlas resolution
% Values:   Continuous probability (0 to 1)
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
    error('create_bnst_atlas_roi:noInput', ...
        'Must provide path to Theiss BNST probability map NIfTI.');
end
if ~exist(atlas_file, 'file')
    error('create_bnst_atlas_roi:fileNotFound', ...
        'Atlas file not found: %s', atlas_file);
end
if nargin < 2 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% Load the BNST probability map
% -------------------------------------------------------------------------
fprintf('Loading Theiss BNST probability map: %s\n', atlas_file);
V_atlas = spm_vol(atlas_file);
Y_atlas = spm_read_vols(V_atlas);

% -------------------------------------------------------------------------
% Normalize to [0, 1] if encoded as percentages
% -------------------------------------------------------------------------
max_val = max(Y_atlas(:));
if max_val > 1
    fprintf('  Atlas values range [0, %.1f]; normalizing to [0, 1].\n', max_val);
    Y_atlas = Y_atlas / max_val;
end

% -------------------------------------------------------------------------
% Threshold at > 0
% -------------------------------------------------------------------------
mask = Y_atlas;
mask(mask <= 0) = 0;

n_nonzero = sum(mask(:) > 0);
mean_prob = mean(mask(mask > 0));
fprintf('  Nonzero voxels: %d (mean probability = %.3f)\n', n_nonzero, mean_prob);

% -------------------------------------------------------------------------
% Write the probabilistic ROI mask
% -------------------------------------------------------------------------
V_out         = V_atlas;
V_out.fname   = fullfile(output_dir, 'roi_atlas_BNST_bilateral.nii');
V_out.dt      = [spm_type('float32') 0];
V_out.pinfo   = [1; 0; 0];
V_out.descrip = 'Bilateral BNST probabilistic ROI from Theiss et al. 2017';

if isfield(V_out, 'private')
    V_out = rmfield(V_out, 'private');
end

V_out = spm_create_vol(V_out);
spm_write_vol(V_out, mask);

fprintf('BNST ROI written: %s\n', V_out.fname);
fprintf('Reference: Theiss et al. NeuroImage 2017;146:288-292\n');
fprintf('Motivation: Lynch et al. Advanced Science 2025 (BNST in pain-arousal circuit)\n');

end
