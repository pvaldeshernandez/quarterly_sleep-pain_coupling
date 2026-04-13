% SCRIPT_STUDY  Master script for the DARTEL preprocessing pipeline.
%
% =========================================================================
% OVERVIEW
% =========================================================================
% This script orchestrates the complete DARTEL (Diffeomorphic Anatomical
% Registration Through Exponentiated Lie Algebra) preprocessing pipeline
% for a BIDS-organized neuroimaging study. It calls a sequence of
% subscripts that perform every step from raw NIfTI images to spatially
% normalized, smoothed volumes ready for group-level statistical analysis.
%
% The pipeline implements the DARTEL approach described in:
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113. doi:10.1016/j.neuroimage.2007.07.007
%
% DARTEL constructs a study-specific template through iterative
% diffeomorphic registration of tissue segmentations, yielding more
% accurate inter-subject alignment than standard SPM normalization,
% particularly for subcortical structures relevant to this study
% (e.g., nucleus accumbens, thalamic nuclei, brainstem).
%
% =========================================================================
% PIPELINE ORDER
% =========================================================================
%
%   Step 0:  exclude_orphan_fMRIs
%            Match functional to structural scans; exclude orphan fMRIs.
%
%   --- Group A (structural stream) ---
%   Step 1:  segment_T1ws
%            Unified segmentation of T1-weighted anatomical images.
%   Step 2:  create_T1w_template
%            Build the study-specific DARTEL template from session-01 T1ws.
%
%   --- Group B (functional stream) ---
%   Step 3:  slice_timing
%            Correct for differences in slice acquisition times.
%   Step 4:  realign_unwarp
%            Realign functional volumes and correct susceptibility-by-
%            movement interactions.
%   Step 5:  coregister_fMIRs_to_T1ws
%            Coregister mean functional images to their T1w anatomicals.
%   Step 6:  segment_fMRIs
%            Segment mean functional images for DARTEL flow-field
%            estimation.
%
%   --- Group C (normalization) ---
%   Step 7:  get_deformations
%            Compute DARTEL flow fields for non-session-01 images by
%            registering to the existing template.
%   Step 8:  deform_all
%            Apply DARTEL deformations (push-forward warping) to all
%            structural and functional images, with optional smoothing.
%   Step 9:  register_T1w_template_to_MNI
%            Bring the DARTEL template into MNI space and create
%            tissue-probability maps at multiple resolutions.
%   Step 10: calculate_averages
%            Compute group-average normalized T1w and fMRI images for
%            quality checking.
%
%   NOTE: Groups A and B are independent and can be run in either order.
%         Group C requires both A and B to be complete.
%
% =========================================================================
% USER-CONFIGURABLE VARIABLES
% =========================================================================
%
%   study       - Short name of the study (used for naming templates and
%                 output files, e.g., 'UPLOAD').
%
%   folder      - Root path to the BIDS directory containing the NIfTI
%                 files. Must end with a path separator. Expected layout:
%                   <folder>/ses-XX/anat/sub-*_ses-XX_T1w.nii
%                   <folder>/ses-XX/func/<task>/sub-*_ses-XX_task-*_bold.nii
%
%   slurmfolder - Path where SLURM job scripts and temporary .mat files
%                 are written. Should be on a fast filesystem.
%
%   ses         - Session label to process (e.g., '01', or '*' for all).
%
%   task        - Functional task label (e.g., 'rest', 'pain', or '*').
%
%   account     - SLURM account/allocation name.
%   burst       - Whether to use burst QOS (1=yes, 0=no).
%   mem         - Memory per job in GB (default: 4 for most steps).
%   time        - Wall-time limit per job (default: '12:00:00').
%   user        - HPC username (for SLURM notifications).
%   matlabver   - MATLAB version string for the SLURM module
%                 (e.g., 'r2018b').
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 (on MATLAB path)
%   - matlab/lib/ (condmkdir, cropImage, reslice_spacen, chckdelslurm)
%   - HiPerGator SLURM utilities (sendslurm, filedir2cell)
%
% =========================================================================

% --- Add project helper functions to path --------------------------------
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'lib'));

% =========================================================================
% USER CONFIGURATION (edit these for your study)
% =========================================================================

% Specify the name of the study
study = '';
% Specify the location of the NIFTI files.
folder = '';
% Specify the SLURM folder.
slurmfolder = '';
% Specify the session (e.g., '01', '*')
ses = '';
% Specify the type of functional task (e.g., 'rest', 'pain', '*')
task = '';

% Set the SLURM parameters
account = 'cruzalmeida';
burst = 1;
mem = 4;
time = '12:00:00';
user = 'pvaldeshernandez';
matlabver = 'r2018b';

% =========================================================================
% PIPELINE EXECUTION
% =========================================================================

% Exclude functional MRIs without structural MRIs and create correpondence
% indices (variable 'fmri2T1wmap')
exclude_orphan_fMRIs

%% Run scripts
% Run Group A of scripts
segment_T1ws
create_T1w_template

% Run Group B of scripts
slice_timing
realign_unwarp
coregister_fMIRs_to_T1ws
segment_fMRIs

% Run Group C of scripts
get_deformations
deform_all
register_T1w_template_to_MNI
calculate_averages

% NOTES:
% The order of execution of Groups A and B is interchangable