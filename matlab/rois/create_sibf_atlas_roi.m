function create_sibf_atlas_roi(atlas_file, output_dir)
% CREATE_SIBF_ATLAS_ROI  Create a probabilistic substantia innominata /
%   basal forebrain Ch4 ROI from the Zaborszky cytoarchitectonic atlas.
%
% Usage:
%   create_sibf_atlas_roi(atlas_file)
%   create_sibf_atlas_roi(atlas_file, output_dir)
%
% Inputs:
%   atlas_file  - Path to the Zaborszky Ch4 probability map NIfTI file.
%                 This is the continuous (0-1 or 0-100%) map of the Ch4
%                 cell group, NOT a binary mask.
%   output_dir  - (Optional) Directory for the output ROI. Default: pwd.
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The substantia innominata (SI) and its cholinergic cell group Ch4
% (nucleus basalis of Meynert) form the primary source of cortical
% cholinergic innervation. Acetylcholine from SI/Ch4 plays a key role
% in cortical arousal, wakefulness maintenance, and pain modulation.
%
% Lynch et al. (2025, Advanced Science) identified the SI/Ch4 as a
% component of the arousal circuit that may modulate pain-to-sleep
% coupling: cholinergic projections from SI/Ch4 to cortex promote
% wakefulness, and disruption of this system could alter how pain
% signals translate into sleep disturbance.
%
% =========================================================================
% ATLAS SOURCE
% =========================================================================
%   Zaborszky L, Hoemke L, Mohlberg H, Schleicher A, Amunts K,
%   Zilles K. Stereotaxic probabilistic maps of the magnocellular
%   cell groups in human basal forebrain. NeuroImage. 2008;42(3):
%   1127-1141. doi:10.1016/j.neuroimage.2008.05.055
%
% The atlas provides continuous probability maps of basal forebrain
% cell groups (Ch1-Ch4 and Ch4p) derived from cytoarchitectonic
% analysis of 10 postmortem brains, registered to MNI152 space.
%
% The Ch4 map gives the probability (0 to 1) that each voxel belongs
% to the Ch4 cell group across the 10 brains. We use this as a
% probabilistic (weighted) mask for ROI extraction: voxels with
% higher probability contribute more to the regional mean.
%
% The atlas is available from the SPM Anatomy Toolbox or from:
%   https://www.fz-juelich.de/en/inm/inm-1/research/atlases/
%
% =========================================================================
% PROCESSING STEPS
% =========================================================================
%   1. Load the Ch4 probability map.
%   2. Threshold at p > 0 to retain any voxel with nonzero probability.
%      This is intentionally liberal because the probability values
%      already encode spatial uncertainty -- they will be used as
%      weights during extraction (probability-weighted mean), so low-
%      probability voxels contribute proportionally little.
%   3. Save the thresholded probability map as the ROI mask.
%
% The output retains continuous probability values (not binarized),
% allowing probability-weighted extraction in extract_atlas_roi_values.m.
%
% =========================================================================
% OUTPUT
% =========================================================================
% Filename: roi_atlas_SIBF_Ch4.nii
% Space:    MNI152, native atlas resolution (typically 1.5 mm or 2 mm)
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
    error('create_sibf_atlas_roi:noInput', ...
        'Must provide path to Zaborszky Ch4 probability map NIfTI.');
end
if ~exist(atlas_file, 'file')
    error('create_sibf_atlas_roi:fileNotFound', ...
        'Atlas file not found: %s', atlas_file);
end
if nargin < 2 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% Load the Ch4 probability map
% -------------------------------------------------------------------------
fprintf('Loading Zaborszky Ch4 probability map: %s\n', atlas_file);
V_atlas = spm_vol(atlas_file);
Y_atlas = spm_read_vols(V_atlas);

% -------------------------------------------------------------------------
% Check value range and normalize to [0, 1] if needed
% -------------------------------------------------------------------------
% Some versions of the atlas encode probabilities as percentages (0-100)
% rather than proportions (0-1). Detect and convert.
max_val = max(Y_atlas(:));
if max_val > 1
    fprintf('  Atlas values range [0, %.1f]; normalizing to [0, 1].\n', max_val);
    Y_atlas = Y_atlas / max_val;
end

% -------------------------------------------------------------------------
% Threshold at > 0 (retain all nonzero-probability voxels)
% -------------------------------------------------------------------------
% We use a minimal threshold rather than a conventional p > 0.25 or
% p > 0.50 cutoff because the probability values will serve as weights
% during ROI extraction. Low-probability voxels contribute
% proportionally little to the weighted mean, so including them does
% not bias the estimate but does improve spatial coverage.
mask = Y_atlas;
mask(mask <= 0) = 0;   % ensure no negative values from interpolation

n_nonzero = sum(mask(:) > 0);
mean_prob = mean(mask(mask > 0));
fprintf('  Nonzero voxels: %d (mean probability = %.3f)\n', n_nonzero, mean_prob);

% -------------------------------------------------------------------------
% Write the probabilistic ROI mask
% -------------------------------------------------------------------------
V_out         = V_atlas;
V_out.fname   = fullfile(output_dir, 'roi_atlas_SIBF_Ch4.nii');
V_out.dt      = [spm_type('float32') 0];   % continuous values need float
V_out.pinfo   = [1; 0; 0];
V_out.descrip = 'SI/BF Ch4 probabilistic ROI from Zaborszky et al. 2008';

if isfield(V_out, 'private')
    V_out = rmfield(V_out, 'private');
end

V_out = spm_create_vol(V_out);
spm_write_vol(V_out, mask);

fprintf('SI/BF Ch4 ROI written: %s\n', V_out.fname);
fprintf('Reference: Zaborszky et al. NeuroImage 2008;42(3):1127-1141\n');

end
