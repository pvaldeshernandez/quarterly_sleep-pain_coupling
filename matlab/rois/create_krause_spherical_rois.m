function create_krause_spherical_rois(output_dir)
% CREATE_KRAUSE_SPHERICAL_ROIS  Build six binary spherical ROI masks from
%   Krause et al. (2019) pain-by-sleep-deprivation coordinates.
%
% Usage:
%   create_krause_spherical_rois           % saves to current directory
%   create_krause_spherical_rois(outdir)   % saves to outdir
%
% =========================================================================
% BACKGROUND
% =========================================================================
% Krause AJ, Prather AA, Wager TD, Lindquist MA, Walker MP. The pain of
% sleep loss: A brain characterization in humans. J Neurosci. 2019;
% 39(12):2291-2300. doi:10.1523/JNEUROSCI.2408-18.2018
%
% In a within-subjects sleep deprivation design (N = 25), Krause et al.
% identified brain regions whose pain-evoked BOLD response changed after
% one night of total sleep deprivation compared with a rested-sleep
% condition. Two classes of effects emerged:
%
%   (1) Somatosensory amplification -- Primary somatosensory cortex (S1)
%       showed INCREASED activation to thermal pain stimuli after sleep
%       deprivation, suggesting that sleep loss amplifies the sensory-
%       discriminative representation of pain.
%
%   (2) Valuation-system blunting -- Striatal (nucleus accumbens),
%       insular, and thalamic regions showed DECREASED activation,
%       suggesting impaired affective evaluation and descending
%       modulation of pain.
%
% These regions form the basis of our sleep-to-pain (SP) coupling
% moderation analysis: we test whether individuals with stronger or
% weaker fMRI responses in these regions show different strengths of
% quarterly sleep-to-pain coupling.
%
% =========================================================================
% ROI DEFINITIONS
% =========================================================================
% Each ROI is defined by an MNI coordinate (center of the sphere) and a
% radius in millimeters, taken from Krause et al. (2019) Table 1 and
% supplementary materials. The radii were chosen to capture the
% approximate spatial extent of each cluster while keeping the ROI
% anatomically contained:
%
%   - 8 mm radius for cortical regions (S1, insula) where clusters were
%     large and spatially diffuse.
%   - 6 mm radius for the nucleus accumbens, a small subcortical
%     structure roughly 10-12 mm in diameter; 6 mm captures most of the
%     nucleus while minimizing overlap with adjacent structures (caudate,
%     ventral pallidum).
%   - 4 mm radius for the thalamus ROI, which targets a specific
%     ventral-lateral nucleus rather than the whole thalamus.
%
%   ROI                 MNI (x,y,z)      Radius  Voxels  SD effect
%   ---                 -----------      ------  ------  ---------
%   Right S1            (36, -31, 59)     8 mm     82    Increased
%   Right Mid Insula    (32,   4, 11)     8 mm     72    Decreased
%   Left Thalamus       (-10, -6, 10)     4 mm     10    Decreased
%   Left Ant Insula     (-27, 25,  0)     8 mm     82    Decreased
%   Left NAcc           (-9,   2, -7)     6 mm     17    Decreased
%   Right NAcc          ( 9,   2, -7)     6 mm     17    Decreased
%
% Note: Left and right NAcc are entered as separate ROIs because
% laterality effects in the NAcc are well documented in chronic pain
% (Makary et al. 2020 PNAS; Egorova-Brumley et al. 2025 Neurobiol Pain).
%
% =========================================================================
% OUTPUT FORMAT
% =========================================================================
% Each mask is a 3D NIfTI-1 file at 3 mm isotropic resolution in MNI152
% space, matching the resolution of the fMRI contrast images. Voxel
% values are binary (0 or 1). The bounding box covers the standard MNI
% brain from (-78,-112,-70) to (78,76,86).
%
% Filenames follow the pattern:
%   roi_krause_<region>_<side>_r<radius>mm.nii
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
% Define MNI grid at 3 mm isotropic resolution
% -------------------------------------------------------------------------
% This bounding box covers the standard MNI152 brain. The voxel-to-world
% affine matrix maps voxel indices (i,j,k) to MNI coordinates (x,y,z):
%
%   [x]   [-3  0  0  78] [i]
%   [y] = [ 0  3  0 -112] [j]
%   [z]   [ 0  0  3 -70] [k]
%   [1]   [ 0  0  0   1] [1]
%
% Grid dimensions: 53 x 63 x 53 voxels
voxel_size = 3;                       % mm, isotropic
origin_mm  = [-78, -112, -70];        % MNI coordinate of voxel (1,1,1)
dim        = [53, 63, 53];            % number of voxels in x, y, z

% Affine transformation: voxel (1,1,1) -> origin_mm
mat = [[-voxel_size 0 0 origin_mm(1) + voxel_size*(dim(1))]; ...
       [0 voxel_size 0 origin_mm(2)]; ...
       [0 0 voxel_size origin_mm(3)]; ...
       [0 0 0 1]];

% Verify: voxel (1,1,1) should map to the positive-x end, and
% voxel (dim(1),1,1) to the negative-x end (radiological convention
% with x-axis flipped). Actually, let's use a cleaner formulation:
%
% We want MNI coordinates to range from -78 to +78 in x, -112 to +76
% in y, and -70 to +86 in z, with 3 mm steps. The affine maps voxel
% index (1-based) to MNI:
%   x = -3*(i-1) + 78    =>  i=1 -> x=78, i=53 -> x=-78
%   y =  3*(j-1) - 112   =>  j=1 -> y=-112, j=63 -> y=74
%   z =  3*(k-1) - 70    =>  k=1 -> z=-70, k=53 -> z=86
%
% In SPM's 4x4 affine (1-indexed):
mat = [-3  0  0  81; ...
        0  3  0 -115; ...
        0  0  3 -73; ...
        0  0  0   1];

% -------------------------------------------------------------------------
% ROI definitions: {name, MNI_xyz, radius_mm, sleep_deprivation_effect}
% -------------------------------------------------------------------------
rois = {
    'roi_krause_S1_right_r8mm',          [ 36, -31, 59],  8, 'increased activation';
    'roi_krause_midInsula_right_r8mm',   [ 32,   4, 11],  8, 'decreased activation';
    'roi_krause_thalamus_left_r4mm',     [-10,  -6, 10],  4, 'decreased activation';
    'roi_krause_antInsula_left_r8mm',    [-27,  25,  0],  8, 'decreased activation';
    'roi_krause_NAcc_left_r6mm',         [ -9,   2, -7],  6, 'decreased activation';
    'roi_krause_NAcc_right_r6mm',        [  9,   2, -7],  6, 'decreased activation';
};

% -------------------------------------------------------------------------
% Create MNI coordinate grids
% -------------------------------------------------------------------------
% For each voxel (i,j,k), compute its MNI (x,y,z) coordinates.
[I, J, K] = ndgrid(1:dim(1), 1:dim(2), 1:dim(3));

% Apply the affine to get MNI coordinates at each voxel
X_mni = mat(1,1)*I + mat(1,2)*J + mat(1,3)*K + mat(1,4);
Y_mni = mat(2,1)*I + mat(2,2)*J + mat(2,3)*K + mat(2,4);
Z_mni = mat(3,1)*I + mat(3,2)*J + mat(3,3)*K + mat(3,4);

% -------------------------------------------------------------------------
% Create each spherical ROI
% -------------------------------------------------------------------------
fprintf('Creating %d Krause spherical ROIs at %d mm resolution...\n', ...
    size(rois,1), voxel_size);

for r = 1:size(rois, 1)
    roi_name   = rois{r, 1};
    center_mni = rois{r, 2};
    radius_mm  = rois{r, 3};
    sd_effect  = rois{r, 4};

    % Compute Euclidean distance from each voxel to the sphere center
    dist = sqrt((X_mni - center_mni(1)).^2 + ...
                (Y_mni - center_mni(2)).^2 + ...
                (Z_mni - center_mni(3)).^2);

    % Threshold: voxels within radius are included (binary mask)
    mask = double(dist <= radius_mm);

    n_voxels = sum(mask(:));

    % Set up SPM volume structure
    V          = struct();
    V.fname    = fullfile(output_dir, [roi_name '.nii']);
    V.dim      = dim;
    V.mat      = mat;
    V.dt       = [spm_type('uint8') 0];   % binary mask, uint8 is sufficient
    V.pinfo    = [1; 0; 0];               % slope=1, intercept=0
    V.n        = [1 1];
    V.descrip  = sprintf('Krause 2019 spherical ROI: %s, center=[%d %d %d], r=%dmm, SD=%s', ...
        strrep(roi_name, 'roi_krause_', ''), ...
        center_mni(1), center_mni(2), center_mni(3), ...
        radius_mm, sd_effect);

    % Write the NIfTI file
    V = spm_create_vol(V);
    spm_write_vol(V, mask);

    fprintf('  %-45s  %3d voxels   center=[%+3d, %+3d, %+3d]  r=%dmm  (%s)\n', ...
        [roi_name '.nii'], n_voxels, ...
        center_mni(1), center_mni(2), center_mni(3), ...
        radius_mm, sd_effect);
end

fprintf('Done. %d ROI masks written to: %s\n', size(rois,1), output_dir);

end
