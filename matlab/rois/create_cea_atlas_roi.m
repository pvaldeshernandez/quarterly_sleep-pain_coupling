function create_cea_atlas_roi(atlas_dir, mni_template, output_dir)
% CREATE_CEA_ATLAS_ROI  Create a probabilistic central nucleus of the
%   amygdala (CeA) ROI from the CIT168 atlas.
%
% Usage:
%   create_cea_atlas_roi(atlas_dir, mni_template)
%   create_cea_atlas_roi(atlas_dir, mni_template, output_dir)
%
% Inputs:
%   atlas_dir    - Directory containing the CIT168 atlas files. Expected
%                  contents include the individual-observer labeling
%                  volumes (e.g., CIT168_*_label_*.nii.gz) and the
%                  CIT168 template image for registration.
%   mni_template - Path to the MNI152 T1 1mm template (e.g., from FSL:
%                  $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz).
%   output_dir   - (Optional) Directory for the output ROI. Default: pwd.
%
% =========================================================================
% BACKGROUND
% =========================================================================
% The central nucleus of the amygdala (CeA) is the primary output
% nucleus of the amygdala and a critical node in the pain-arousal
% circuit. It receives direct projections from the parabrachial nucleus
% (PBN) via the PBN^elCGRP pathway and projects to arousal-promoting
% regions including the locus coeruleus and lateral hypothalamus.
%
% Lynch et al. (2025, Advanced Science) identified the CeA as the
% first forebrain relay in the ascending pain-induced wakefulness
% pathway: nociceptive signals from PBN^elCGRP neurons synapse in the
% CeA, which then activates downstream arousal systems. This makes
% the CeA a candidate moderator of pain-to-sleep (PS) coupling.
%
% =========================================================================
% ATLAS SOURCE
% =========================================================================
%   Pauli WM, Nili AN, Tyszka JM. A high-resolution probabilistic
%   in vivo atlas of human subcortical brain nuclei. Scientific Data.
%   2018;5:180063. doi:10.1038/sdata.2018.63
%
% The CIT168 atlas provides probabilistic labels for subcortical nuclei
% derived from crowd-sourced manual segmentations. For each nucleus,
% 2 expert observers independently labeled 8 template brains (the
% CIT168 high-resolution T1 templates), yielding 16 labelings per
% nucleus. The probability map for each nucleus is the voxelwise mean
% across all 16 labelings.
%
% The CeA label is:
%   Label 4: AMY_CEN (central nucleus of amygdala)
%
% Note: The CIT168 atlas is defined in its own template space, NOT in
% standard MNI152. Registration to MNI152 is required.
%
% The atlas is freely available (CC-BY-4.0 license) from:
%   https://osf.io/jkzwp/ (CIT168 Subcortical Atlas)
%
% =========================================================================
% PROCESSING STEPS
% =========================================================================
%   1. Load all 16 individual-observer labeling volumes.
%   2. For each volume, extract binary mask where label == 4 (AMY_CEN).
%   3. Average the 16 binary masks to create a probability map (0 to 1).
%   4. Register the CIT168 template to MNI152 using:
%      a. FSL FLIRT: 12-parameter affine (initial alignment)
%      b. FSL FNIRT: nonlinear warp (fine alignment)
%   5. Apply the combined warp to the probability map using FSL
%      applywarp with trilinear interpolation.
%   6. Mirror the warped map bilaterally (if the atlas provides only
%      unilateral labels).
%   7. Save the probability map in MNI152 space.
%
% This multi-step registration is necessary because the CIT168 template
% was constructed independently of MNI152, and simple affine alignment
% is insufficient for subcortical structures (Pauli et al. 2018).
%
% =========================================================================
% OUTPUT
% =========================================================================
% Filename: roi_atlas_CeA_bilateral.nii
% Space:    MNI152, 1 mm isotropic
% Values:   Continuous probability (0 to 1)
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (for spm_vol, spm_read_vols, spm_create_vol, spm_write_vol)
%   - FSL  (for flirt, fnirt, applywarp; must be on system PATH)
%
% =========================================================================
% Author: Pedro Valdes-Hernandez, University of Florida, 2025-2026
% =========================================================================

% -------------------------------------------------------------------------
% Input validation
% -------------------------------------------------------------------------
if nargin < 2
    error('create_cea_atlas_roi:missingInputs', ...
        'Must provide atlas_dir and mni_template.');
end
if ~exist(atlas_dir, 'dir')
    error('create_cea_atlas_roi:dirNotFound', ...
        'Atlas directory not found: %s', atlas_dir);
end
if ~exist(mni_template, 'file')
    error('create_cea_atlas_roi:templateNotFound', ...
        'MNI template not found: %s', mni_template);
end
if nargin < 3 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% Constants
% -------------------------------------------------------------------------
CEA_LABEL = 4;           % AMY_CEN in CIT168 labeling scheme
N_OBSERVERS = 2;         % number of expert observers
N_TEMPLATES = 8;         % number of CIT168 template brains
N_LABELINGS = N_OBSERVERS * N_TEMPLATES;   % = 16 total labelings

% -------------------------------------------------------------------------
% Step 1: Locate individual-observer labeling files
% -------------------------------------------------------------------------
fprintf('Step 1: Loading %d individual-observer labelings from CIT168...\n', ...
    N_LABELINGS);

% The CIT168 atlas distributes individual labelings as separate NIfTI
% files. The naming convention may vary; we search for all label files.
label_files = dir(fullfile(atlas_dir, 'CIT168_*_label_*.nii*'));
if isempty(label_files)
    % Try alternative naming convention
    label_files = dir(fullfile(atlas_dir, '*label*.nii*'));
end
if numel(label_files) < N_LABELINGS
    warning('create_cea_atlas_roi:fewLabelings', ...
        'Found %d labeling files (expected %d). Using available files.', ...
        numel(label_files), N_LABELINGS);
end

% -------------------------------------------------------------------------
% Steps 2-3: Extract CeA from each labeling and average
% -------------------------------------------------------------------------
fprintf('Step 2-3: Extracting label %d (AMY_CEN) and computing probability map...\n', ...
    CEA_LABEL);

% Load the first file to get dimensions
V_ref = spm_vol(fullfile(label_files(1).folder, label_files(1).name));
prob_map = zeros(V_ref.dim);
n_files = numel(label_files);

for f = 1:n_files
    fpath = fullfile(label_files(f).folder, label_files(f).name);
    V_tmp = spm_vol(fpath);
    Y_tmp = spm_read_vols(V_tmp);

    % Binary mask: 1 where label equals CeA, 0 elsewhere
    prob_map = prob_map + double(Y_tmp == CEA_LABEL);
end

% Divide by number of labelings to get probability (0 to 1)
prob_map = prob_map / n_files;

n_nonzero = sum(prob_map(:) > 0);
fprintf('  CeA probability map: %d nonzero voxels (from %d labelings)\n', ...
    n_nonzero, n_files);

% Save the probability map in CIT168 space (intermediate file)
cea_cit168 = fullfile(output_dir, 'temp_cea_cit168_prob.nii');
V_prob         = V_ref;
V_prob.fname   = cea_cit168;
V_prob.dt      = [spm_type('float32') 0];
V_prob.pinfo   = [1; 0; 0];
V_prob.descrip = 'CeA probability map in CIT168 space (intermediate)';
if isfield(V_prob, 'private')
    V_prob = rmfield(V_prob, 'private');
end
V_prob = spm_create_vol(V_prob);
spm_write_vol(V_prob, prob_map);

% -------------------------------------------------------------------------
% Step 4: Register CIT168 template to MNI152 via FLIRT + FNIRT
% -------------------------------------------------------------------------
fprintf('Step 4: Registering CIT168 template to MNI152...\n');

% Locate the CIT168 template (the structural image used for atlas
% construction, needed as the "moving" image for registration)
cit168_template_files = dir(fullfile(atlas_dir, 'CIT168_T1w*.nii*'));
if isempty(cit168_template_files)
    cit168_template_files = dir(fullfile(atlas_dir, 'CIT168_template*.nii*'));
end
if isempty(cit168_template_files)
    error('create_cea_atlas_roi:noTemplate', ...
        'CIT168 template image not found in: %s', atlas_dir);
end
cit168_template = fullfile(cit168_template_files(1).folder, ...
    cit168_template_files(1).name);

% Intermediate files for registration
affine_mat  = fullfile(output_dir, 'temp_cit168_to_mni_affine.mat');
warp_coeff  = fullfile(output_dir, 'temp_cit168_to_mni_warp.nii.gz');
cea_mni     = fullfile(output_dir, 'temp_cea_mni_prob.nii.gz');

% Step 4a: Affine registration with FLIRT (12 DOF)
fprintf('  4a: FLIRT affine registration (12 DOF)...\n');
cmd_flirt = sprintf(['flirt -in %s -ref %s -omat %s ' ...
    '-dof 12 -cost corratio -searchrx -30 30 ' ...
    '-searchry -30 30 -searchrz -30 30'], ...
    cit168_template, mni_template, affine_mat);
[status, result] = system(cmd_flirt);
if status ~= 0
    error('create_cea_atlas_roi:flirtFailed', ...
        'FLIRT failed:\n%s', result);
end

% Step 4b: Nonlinear registration with FNIRT
fprintf('  4b: FNIRT nonlinear registration...\n');
cmd_fnirt = sprintf(['fnirt --in=%s --ref=%s --aff=%s ' ...
    '--cout=%s --iout=%s'], ...
    cit168_template, mni_template, affine_mat, ...
    warp_coeff, fullfile(output_dir, 'temp_cit168_warped.nii.gz'));
[status, result] = system(cmd_fnirt);
if status ~= 0
    error('create_cea_atlas_roi:fnirtFailed', ...
        'FNIRT failed:\n%s', result);
end

% -------------------------------------------------------------------------
% Step 5: Apply the warp to the CeA probability map
% -------------------------------------------------------------------------
fprintf('Step 5: Applying warp to CeA probability map...\n');

cmd_applywarp = sprintf(['applywarp --in=%s --ref=%s --warp=%s ' ...
    '--out=%s --interp=trilinear'], ...
    cea_cit168, mni_template, warp_coeff, cea_mni);
[status, result] = system(cmd_applywarp);
if status ~= 0
    error('create_cea_atlas_roi:applywarpFailed', ...
        'applywarp failed:\n%s', result);
end

% -------------------------------------------------------------------------
% Step 6: Mirror bilaterally (if needed)
% -------------------------------------------------------------------------
fprintf('Step 6: Creating bilateral mask by mirroring across midline...\n');

V_mni = spm_vol(cea_mni);
Y_mni = spm_read_vols(V_mni);

% The MNI x-axis is encoded in the first row of the affine. We flip the
% volume along the x-dimension to create a mirrored copy, then take the
% voxelwise maximum of the original and mirrored maps.
Y_flipped = Y_mni(end:-1:1, :, :);

% Take the maximum: if a voxel has nonzero probability in either the
% original or mirrored map, it is included
Y_bilateral = max(Y_mni, Y_flipped);

% Clip to [0, 1] (interpolation can produce small overshoots)
Y_bilateral = max(0, min(1, Y_bilateral));

n_bilateral = sum(Y_bilateral(:) > 0);
fprintf('  Bilateral CeA: %d nonzero voxels\n', n_bilateral);

% -------------------------------------------------------------------------
% Step 7: Save the final probabilistic ROI in MNI152 space
% -------------------------------------------------------------------------
V_out         = V_mni;
V_out.fname   = fullfile(output_dir, 'roi_atlas_CeA_bilateral.nii');
V_out.dt      = [spm_type('float32') 0];
V_out.pinfo   = [1; 0; 0];
V_out.descrip = 'Bilateral CeA probabilistic ROI from CIT168 (Pauli et al. 2018), MNI152 space';

if isfield(V_out, 'private')
    V_out = rmfield(V_out, 'private');
end

V_out = spm_create_vol(V_out);
spm_write_vol(V_out, Y_bilateral);

fprintf('CeA ROI written: %s\n', V_out.fname);

% -------------------------------------------------------------------------
% Clean up intermediate files
% -------------------------------------------------------------------------
fprintf('Cleaning up intermediate files...\n');
temp_files = dir(fullfile(output_dir, 'temp_*'));
for t = 1:numel(temp_files)
    delete(fullfile(temp_files(t).folder, temp_files(t).name));
end

fprintf('Reference: Pauli et al. Scientific Data 2018;5:180063\n');
fprintf('Motivation: Lynch et al. Advanced Science 2025 (CeA in pain-arousal circuit)\n');

end
