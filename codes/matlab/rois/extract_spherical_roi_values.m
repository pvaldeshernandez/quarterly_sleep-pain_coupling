function values = extract_spherical_roi_values(con_images, roi_mask)
% EXTRACT_SPHERICAL_ROI_VALUES  Extract mean BOLD contrast values within
%   a binary spherical ROI for each subject's contrast image.
%
% Usage:
%   values = extract_spherical_roi_values(con_images, roi_mask)
%
% Inputs:
%   con_images  - Cell array of file paths to subject-level contrast
%                 images (con_0001.nii), OR a character array where each
%                 row is a file path.
%   roi_mask    - File path to the binary ROI mask NIfTI file (created by
%                 create_krause_spherical_rois.m or create_acc_spherical_roi.m).
%
% Output:
%   values      - N x 1 vector of mean BOLD contrast values, where N is
%                 the number of subjects. NaN entries indicate subjects
%                 for whom no valid data existed within the ROI.
%
% =========================================================================
% FORMULA
% =========================================================================
% For each subject i, the extracted value is the unweighted mean of all
% valid voxels within the spherical ROI:
%
%   X_bar_i = (1 / K_i) * sum_{v in ROI} BOLD_iv
%
% where:
%   BOLD_iv = contrast estimate at voxel v for subject i
%   K_i     = number of non-NaN voxels within the ROI for subject i
%
% This is equivalent to nanmean(BOLD(ROI_voxels)) for each subject.
%
% =========================================================================
% WHY nanmean IS NEEDED
% =========================================================================
% Some contrast images were estimated with individual gray matter (GM)
% masks in SPM's first-level model. Voxels outside a subject's GM mask
% are set to NaN (or zero, depending on SPM version). For subcortical
% ROIs like the nucleus accumbens or thalamus, the GM mask may not cover
% all voxels in the spherical ROI (e.g., the left thalamus 4mm-radius
% sphere has only 10 voxels, and 2 might fall outside a given subject's
% GM mask).
%
% Using nanmean ensures that:
%   (1) Subjects with partial coverage still contribute to the analysis
%       (the mean is computed over available voxels only).
%   (2) Subjects with NO valid voxels in the ROI return NaN, which
%       downstream code can handle (e.g., exclude from moderation
%       analysis).
%
% Note: For the "no-mask" re-estimated contrasts (see
% matlab/nomask_reestimation/), all voxels are valid and nanmean
% behaves identically to mean.
%
% =========================================================================
% ASSUMPTIONS
% =========================================================================
%   - The contrast images and ROI mask are in the same space (MNI152)
%     and at the same voxel resolution (3 mm isotropic for fMRI).
%   - The ROI mask is binary (0 or 1). For probability-weighted
%     extraction, use extract_atlas_roi_values.m instead.
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (for spm_vol, spm_read_vols)
%
% =========================================================================
% Author: Pedro Valdes-Hernandez, University of Florida, 2025-2026
% =========================================================================

% -------------------------------------------------------------------------
% Input handling
% -------------------------------------------------------------------------
% Convert character array to cell array for uniform processing
if ischar(con_images)
    con_images = cellstr(con_images);
end
n_subjects = numel(con_images);

% -------------------------------------------------------------------------
% Load the ROI mask
% -------------------------------------------------------------------------
V_roi = spm_vol(roi_mask);
Y_roi = spm_read_vols(V_roi);

% Find voxel indices within the ROI (nonzero voxels)
roi_idx = find(Y_roi(:) > 0);
n_roi_voxels = numel(roi_idx);

if n_roi_voxels == 0
    error('extract_spherical_roi_values:emptyROI', ...
        'ROI mask contains no nonzero voxels: %s', roi_mask);
end

fprintf('Extracting from ROI: %s (%d voxels)\n', roi_mask, n_roi_voxels);

% -------------------------------------------------------------------------
% Verify spatial compatibility with the first contrast image
% -------------------------------------------------------------------------
V_check = spm_vol(con_images{1});
if ~isequal(V_check.dim, V_roi.dim)
    error('extract_spherical_roi_values:dimMismatch', ...
        ['Dimension mismatch between ROI mask (%s) and contrast image (%s).\n' ...
         'ROI: [%d %d %d], Image: [%d %d %d].\n' ...
         'Ensure both are at the same resolution (e.g., 3mm isotropic).'], ...
        roi_mask, con_images{1}, ...
        V_roi.dim(1), V_roi.dim(2), V_roi.dim(3), ...
        V_check.dim(1), V_check.dim(2), V_check.dim(3));
end

% -------------------------------------------------------------------------
% Extract values for each subject
% -------------------------------------------------------------------------
values = nan(n_subjects, 1);

for s = 1:n_subjects
    % Load the contrast image
    V_con = spm_vol(con_images{s});
    Y_con = spm_read_vols(V_con);

    % Extract voxel values within the ROI
    roi_values = Y_con(roi_idx);

    % Compute the unweighted mean, ignoring NaN voxels
    % (NaN arises from GM-masked contrast images; see header)
    valid = ~isnan(roi_values) & ~isinf(roi_values);
    n_valid = sum(valid);

    if n_valid > 0
        values(s) = mean(roi_values(valid));
    else
        % No valid voxels: this subject has no data in this ROI
        values(s) = NaN;
        warning('extract_spherical_roi_values:noValidVoxels', ...
            'Subject %d (%s): 0 valid voxels in ROI (returning NaN).', ...
            s, con_images{s});
    end
end

% -------------------------------------------------------------------------
% Summary statistics
% -------------------------------------------------------------------------
n_valid_subjects = sum(~isnan(values));
n_nan_subjects   = sum(isnan(values));
fprintf('  Extracted: %d subjects (%d valid, %d NaN)\n', ...
    n_subjects, n_valid_subjects, n_nan_subjects);
if n_valid_subjects > 0
    fprintf('  Mean = %.4f, SD = %.4f, range = [%.4f, %.4f]\n', ...
        nanmean(values), nanstd(values), ...
        min(values(~isnan(values))), max(values(~isnan(values))));
end

end
