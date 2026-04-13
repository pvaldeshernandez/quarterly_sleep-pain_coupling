function [bold_values, gm_volumes] = extract_atlas_roi_values(con_images, smwc1_images, atlas_mask)
% EXTRACT_ATLAS_ROI_VALUES  Extract probability-weighted BOLD and GM
%   volume values from atlas-defined ROIs.
%
% Usage:
%   [bold_values, gm_volumes] = extract_atlas_roi_values(con_images, smwc1_images, atlas_mask)
%   bold_values = extract_atlas_roi_values(con_images, [], atlas_mask)
%
% Inputs:
%   con_images    - Cell array of file paths to subject-level contrast
%                   images (con_0001.nii). Pass [] to skip BOLD extraction.
%   smwc1_images  - Cell array of file paths to smoothed, modulated,
%                   warped GM segmentations (smwc1*.nii). These are the
%                   VBM images from the DARTEL pipeline. Pass [] to skip
%                   GM volume extraction.
%   atlas_mask    - File path to the probabilistic atlas ROI NIfTI file
%                   (created by one of the create_*_atlas_roi.m scripts).
%                   Values should be in [0, 1], representing the
%                   probability that each voxel belongs to the ROI.
%
% Outputs:
%   bold_values   - N x 1 vector of probability-weighted mean BOLD
%                   contrast values. Empty if con_images is [].
%   gm_volumes    - N x 1 vector of probability-weighted GM volumes
%                   in mm^3. Empty if smwc1_images is [].
%
% =========================================================================
% FORMULAS
% =========================================================================
%
% --- Probability-weighted mean BOLD ---
%
%   X_bar_i = sum_v( BOLD_iv * w_v ) / sum_v( w_v )
%
% where:
%   BOLD_iv = contrast estimate at voxel v for subject i
%   w_v     = atlas probability at voxel v (0 to 1)
%   sum_v   = sum over all voxels where w_v > 0
%
% This is a weighted mean where voxels with higher probability of
% belonging to the ROI contribute more. For binary atlases (w_v in
% {0, 1}), this reduces to the unweighted mean (equivalent to
% extract_spherical_roi_values.m).
%
% Note: Only voxels where BOTH w_v > 0 AND BOLD_iv is not NaN
% contribute. The denominator is adjusted accordingly:
%
%   X_bar_i = sum_v( BOLD_iv * w_v * valid_v ) / sum_v( w_v * valid_v )
%
% where valid_v = 1 if BOLD_iv is finite, 0 otherwise.
%
% --- Probability-weighted GM volume ---
%
%   Vol_i = sum_v( GM_iv * w_v * V_vox )
%
% where:
%   GM_iv  = smoothed, modulated GM concentration at voxel v for
%            subject i (from smwc1 images). After Jacobian modulation,
%            the voxel value represents the amount of GM "packed into"
%            that voxel — integrating over space gives total GM volume.
%   w_v    = atlas probability at voxel v
%   V_vox  = volume of one voxel in mm^3
%
% For 1.5 mm isotropic VBM images: V_vox = 1.5^3 = 3.375 mm^3.
%
% The result is in mm^3 and represents the total gray matter volume
% within the atlas-defined ROI, weighted by the atlas probability at
% each voxel. This is the standard approach for atlas-based volumetry
% from VBM (e.g., Eickhoff et al. 2005 NeuroImage).
%
% =========================================================================
% RESAMPLING
% =========================================================================
% The atlas mask and the subject images may be at different resolutions:
%   - fMRI contrast images are at 3 mm isotropic (DARTEL-normalized)
%   - VBM smwc1 images are at 1.5 mm isotropic (DARTEL-normalized)
%   - Atlas masks vary (0.5 mm to 2 mm depending on the atlas)
%
% Before extraction, the atlas mask is resampled to the target image
% grid using SPM's reslicing with trilinear interpolation. This ensures
% that the mask and data are in register at the voxel level. Trilinear
% interpolation is appropriate for continuous probability maps. After
% resampling, small negative values from interpolation are clipped to 0.
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (for spm_vol, spm_read_vols, spm_create_vol, spm_write_vol,
%            spm_reslice)
%
% =========================================================================
% Author: Pedro Valdes-Hernandez, University of Florida, 2025-2026
% =========================================================================

% -------------------------------------------------------------------------
% Input handling
% -------------------------------------------------------------------------
do_bold = ~isempty(con_images);
do_gm   = ~isempty(smwc1_images);

if ~do_bold && ~do_gm
    error('extract_atlas_roi_values:noImages', ...
        'At least one of con_images or smwc1_images must be provided.');
end
if ~exist(atlas_mask, 'file')
    error('extract_atlas_roi_values:maskNotFound', ...
        'Atlas mask not found: %s', atlas_mask);
end

% Convert char to cell if needed
if ischar(con_images),    con_images = cellstr(con_images);       end
if ischar(smwc1_images),  smwc1_images = cellstr(smwc1_images);   end

% Determine number of subjects
if do_bold
    n_subjects = numel(con_images);
elseif do_gm
    n_subjects = numel(smwc1_images);
end

% Verify consistent subject counts
if do_bold && do_gm
    if numel(con_images) ~= numel(smwc1_images)
        error('extract_atlas_roi_values:subjectMismatch', ...
            'Number of contrast images (%d) ~= number of smwc1 images (%d).', ...
            numel(con_images), numel(smwc1_images));
    end
end

% Initialize outputs
bold_values = [];
gm_volumes  = [];

% -------------------------------------------------------------------------
% BOLD extraction
% -------------------------------------------------------------------------
if do_bold
    fprintf('--- BOLD extraction (probability-weighted mean) ---\n');
    bold_values = extract_weighted_values(con_images, atlas_mask, 'bold');
end

% -------------------------------------------------------------------------
% GM volume extraction
% -------------------------------------------------------------------------
if do_gm
    fprintf('--- GM volume extraction (probability-weighted integral) ---\n');
    gm_volumes = extract_weighted_values(smwc1_images, atlas_mask, 'volume');
end

end


% =========================================================================
% HELPER FUNCTION: Resample atlas mask and extract weighted values
% =========================================================================
function values = extract_weighted_values(image_files, atlas_mask, mode)
% EXTRACT_WEIGHTED_VALUES  Internal helper that resamples the atlas mask
%   to the target image resolution and extracts probability-weighted
%   values (mean for 'bold' mode, integral for 'volume' mode).

n_subjects = numel(image_files);
values = nan(n_subjects, 1);

% Load atlas mask at native resolution
V_atlas = spm_vol(atlas_mask);
Y_atlas = spm_read_vols(V_atlas);

% Load first subject image to get target grid
V_target = spm_vol(image_files{1});

% Check if resampling is needed (compare dimensions and affine)
needs_resample = ~isequal(V_atlas.dim, V_target.dim) || ...
                 max(abs(V_atlas.mat(:) - V_target.mat(:))) > 0.01;

if needs_resample
    fprintf('  Resampling atlas mask from [%d %d %d] to [%d %d %d]...\n', ...
        V_atlas.dim(1), V_atlas.dim(2), V_atlas.dim(3), ...
        V_target.dim(1), V_target.dim(2), V_target.dim(3));

    % Resample atlas to target grid using SPM's interpolation.
    % We create a temporary file, reslice, then load and clean up.
    %
    % Alternative approach: sample atlas at each target voxel's MNI
    % coordinate using the inverse affine. This avoids temp files.
    Y_resampled = resample_volume(V_atlas, Y_atlas, V_target);
else
    Y_resampled = Y_atlas;
end

% Clip to [0, 1] (interpolation artifacts)
Y_resampled = max(0, min(1, Y_resampled));

% Weights: the atlas probability at each voxel
weights = Y_resampled(:);
roi_idx = find(weights > 0);
n_roi_voxels = numel(roi_idx);
w_roi = weights(roi_idx);   % nonzero weights only

if n_roi_voxels == 0
    error('extract_atlas_roi_values:emptyROI', ...
        'Atlas mask contains no nonzero voxels after resampling: %s', ...
        atlas_mask);
end

fprintf('  ROI: %d nonzero voxels (sum weights = %.2f)\n', ...
    n_roi_voxels, sum(w_roi));

% Compute voxel volume in mm^3 for volume mode
voxel_size = abs(diag(V_target.mat(1:3, 1:3)));
V_vox = prod(voxel_size);   % mm^3 per voxel

for s = 1:n_subjects
    V_sub = spm_vol(image_files{s});
    Y_sub = spm_read_vols(V_sub);

    % Extract ROI voxel values
    sub_values = Y_sub(roi_idx);

    % Identify valid (non-NaN, non-Inf) voxels
    valid = isfinite(sub_values);

    if sum(valid) == 0
        values(s) = NaN;
        warning('extract_atlas_roi_values:noValidVoxels', ...
            'Subject %d: 0 valid voxels in ROI (returning NaN).', s);
        continue;
    end

    switch mode
        case 'bold'
            % Probability-weighted mean BOLD:
            %   X_bar = sum(BOLD * w) / sum(w)  [over valid voxels]
            values(s) = sum(sub_values(valid) .* w_roi(valid)) / ...
                        sum(w_roi(valid));

        case 'volume'
            % Probability-weighted GM volume:
            %   Vol = sum(GM * w * V_vox)  [over valid voxels]
            values(s) = sum(sub_values(valid) .* w_roi(valid)) * V_vox;
    end
end

% Summary
n_valid = sum(~isnan(values));
n_nan   = sum(isnan(values));
fprintf('  Subjects: %d total, %d valid, %d NaN\n', ...
    n_subjects, n_valid, n_nan);
if n_valid > 0
    fprintf('  Mean = %.4f, SD = %.4f\n', ...
        nanmean(values), nanstd(values));
    if strcmp(mode, 'volume')
        fprintf('  (Values in mm^3; voxel volume = %.3f mm^3)\n', V_vox);
    end
end

end


% =========================================================================
% HELPER FUNCTION: Resample a volume to a target grid
% =========================================================================
function Y_out = resample_volume(V_source, Y_source, V_target)
% RESAMPLE_VOLUME  Resample Y_source (defined by V_source affine) onto
%   the voxel grid defined by V_target, using trilinear interpolation.
%
% This implements the resampling without writing temporary files:
%   For each voxel (i,j,k) in the target grid, compute its MNI
%   coordinate using V_target.mat, then find the corresponding
%   fractional voxel in the source grid using inv(V_source.mat),
%   and interpolate.

% Target grid dimensions
dim_out = V_target.dim(1:3);
Y_out = zeros(dim_out);

% Combined transformation: target voxel -> source voxel
% target_voxel -> MNI: V_target.mat
% MNI -> source_voxel: inv(V_source.mat)
M = V_source.mat \ V_target.mat;

% Create target voxel index grids (1-based)
[I, J, K] = ndgrid(1:dim_out(1), 1:dim_out(2), 1:dim_out(3));

% Map to source voxel coordinates (fractional)
X_src = M(1,1)*I + M(1,2)*J + M(1,3)*K + M(1,4);
Y_src = M(2,1)*I + M(2,2)*J + M(2,3)*K + M(2,4);
Z_src = M(3,1)*I + M(3,2)*J + M(3,3)*K + M(3,4);

% Trilinear interpolation using MATLAB's interp3
% Note: interp3 uses (Y, X, Z) ordering (column, row, slice)
dim_src = size(Y_source);
Y_out = interp3(...
    single(Y_source), ...
    single(Y_src), ...     % columns in source (maps to 2nd dim)
    single(X_src), ...     % rows in source (maps to 1st dim)
    single(Z_src), ...     % slices in source (maps to 3rd dim)
    'linear', 0);          % 0 for out-of-bounds

Y_out = double(Y_out);

end
