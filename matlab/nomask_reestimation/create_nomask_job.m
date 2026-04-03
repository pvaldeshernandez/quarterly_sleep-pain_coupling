function job = create_nomask_job(spm_mat_path, output_dir)
% CREATE_NOMASK_JOB  Prepare a re-estimation job without the GM mask.
%
% job = create_nomask_job(spm_mat_path, output_dir)
%
% =========================================================================
% WHY THIS IS NEEDED
% =========================================================================
% SPM's first-level GLM was originally estimated with an individual grey-
% matter (GM) mask derived from each participant's segmented T1w. Every
% voxel outside that mask was set to NaN in the contrast image. This is
% fine for cortical analyses, but subcortical ROIs lose critical voxels
% because the GM probability is low there:
%
%   - Left thalamus (4 mm sphere, 10 voxels): only 2 survived the mask
%   - Parabrachial nucleus (PBN):             0 voxels at 3 mm resolution
%
% The fix is simple: re-estimate the same GLM contrast using the FULL
% brain (every voxel with nonzero signal across time) instead of just the
% GM-masked voxels. No model parameters change; only the spatial extent of
% the output differs.
%
% =========================================================================
% THE OLS RE-ESTIMATION PROCEDURE
% =========================================================================
% SPM stores everything needed to reproduce the beta estimates from the
% original GLM in the SPM.mat file. The key objects are:
%
%   SPM.xX.X      - Design matrix X               [n_scans x n_regressors]
%   SPM.xX.W      - Whitening matrix W             [n_scans x n_scans]
%   SPM.xX.K      - High-pass filter structure     (128 s DCT basis set)
%   SPM.xX.pKX    - Precomputed pseudoinverse       pinv(W * K * X)
%   SPM.xGX.gSF   - Global scaling factors         [n_scans x 1]
%   SPM.xCon(1).c - Contrast vector                [n_regressors x 1]
%   SPM.xY.VY     - spm_vol headers for the 4D functional data
%
% For any voxel time series y (n_scans x 1), the preprocessing pipeline is:
%
%   1. Global scaling:   y_scaled = y .* gSF
%   2. High-pass filter: y_filt   = y_scaled - X0 * pinv(X0) * y_scaled
%                         where X0 is the DCT basis set from SPM.xX.K
%   3. Whitening:        y_white  = W * y_filt
%
% Then the OLS solution is:
%
%   beta = pKX * y_white        [n_regressors x 1]
%   con  = c' * beta            [scalar]
%
% where pKX = pinv(W * K * X), precomputed by SPM and stored in the .mat.
%
% This function reads all these ingredients from SPM.mat and packages them
% into a struct. It does NOT load the functional data or run anything.
%
% =========================================================================
% VALIDATION
% =========================================================================
% The Python implementation of this same algorithm was validated against
% SPM's original masked estimates. Within the mask, voxel-wise correlation
% between original and re-estimated contrast images was r = 0.99999.
%
% Reference: manuscript Methods, "MRI preprocessing and first-level
% analysis", paragraph on masking.
%
% =========================================================================
% INPUT
% =========================================================================
%   spm_mat_path - Full path to SPM.mat (char or string)
%   output_dir   - Directory where con_0001.nii will be saved (char or
%                  string). Created if it does not exist.
%
% OUTPUT
%   job          - Struct with fields:
%       .spm_mat_path  - Path to the SPM.mat that was loaded
%       .output_dir    - Output directory (created if needed)
%       .output_file   - Full path to the output con_0001.nii
%       .pKX           - Precomputed pseudoinverse  [n_reg x n_scans]
%       .gSF           - Global scaling factors     [n_scans x 1]
%       .X0            - DCT high-pass filter basis [n_scans x n_dct]
%       .X0_pinv       - Pseudoinverse of X0        [n_dct x n_scans]
%       .W             - Whitening matrix           [n_scans x n_scans]
%       .c             - Contrast vector            [n_reg x 1]
%       .func_file     - Path to the 4D functional NIfTI
%       .n_scans       - Number of time points (volumes)
%       .n_regressors  - Number of columns in the design matrix
%
% =========================================================================
% DEPENDENCIES
% =========================================================================
%   - SPM12 on the MATLAB path (for spm_vol in run_nomask_reestimation)
%
% See also: run_nomask_reestimation

% -------------------------------------------------------------------------
% Validate inputs
% -------------------------------------------------------------------------
spm_mat_path = char(spm_mat_path);
output_dir   = char(output_dir);

if ~exist(spm_mat_path, 'file')
    error('create_nomask_job:fileNotFound', ...
          'SPM.mat not found: %s', spm_mat_path);
end

% Create output directory if it does not exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% -------------------------------------------------------------------------
% Load SPM.mat
% -------------------------------------------------------------------------
% SPM.mat contains the full specification of the first-level GLM:
% design matrix, whitening, filtering, scaling, contrast definitions, and
% file handles for the functional data.
tmp = load(spm_mat_path, 'SPM');
SPM = tmp.SPM;

% -------------------------------------------------------------------------
% Extract the precomputed pseudoinverse pKX
% -------------------------------------------------------------------------
% pKX = pinv(W * K * X), where K is the high-pass filtering operator.
% SPM precomputes this at estimation time. It is an [n_reg x n_scans]
% matrix that maps whitened, filtered data directly to beta estimates.
pKX = SPM.xX.pKX;

% If stored as sparse (possible in some SPM versions), convert to full
if issparse(pKX)
    pKX = full(pKX);
end

[n_regressors, n_scans] = size(pKX);

% -------------------------------------------------------------------------
% Extract global scaling factors
% -------------------------------------------------------------------------
% SPM applies proportional scaling so that each volume's global mean is
% brought to a common value. gSF is the multiplicative factor for each
% scan. It is applied element-wise to the voxel time series BEFORE
% filtering and whitening.
gSF = SPM.xGX.gSF(:);  % force column vector [n_scans x 1]

if length(gSF) ~= n_scans
    error('create_nomask_job:sizeMismatch', ...
          'gSF has %d elements but pKX expects %d scans.', ...
          length(gSF), n_scans);
end

% -------------------------------------------------------------------------
% Extract the high-pass filter (128 s DCT basis set)
% -------------------------------------------------------------------------
% SPM's high-pass filter is implemented as a residual-forming matrix:
%
%   K(y) = y - X0 * pinv(X0) * y
%
% where X0 is a discrete cosine transform (DCT) basis set whose columns
% span all frequencies below 1/128 Hz. Subtracting the projection onto X0
% removes slow drifts (scanner drift, physiological low-frequency noise).
%
% SPM stores the DCT basis in SPM.xX.K.X0 (the field name "X0" is SPM's
% convention for the "confound" part of the filter). For a single-session
% design, K is a 1x1 struct. For multi-session designs, K is a struct
% array with one element per session (we handle both cases).

K = SPM.xX.K;

if length(K) == 1
    % Single session — the usual case for our task fMRI data
    X0 = K.X0;
else
    % Multi-session — build a block-diagonal X0
    % Each session has its own DCT basis; they are concatenated along the
    % diagonal so that each session's filter only affects its own scans.
    X0 = blkdiag(K.X0);
end

% Precompute the pseudoinverse of X0. This is used in the filtering step:
%   y_filtered = y - X0 * X0_pinv * y
X0_pinv = pinv(X0);

% -------------------------------------------------------------------------
% Extract the whitening matrix
% -------------------------------------------------------------------------
% W decorrelates the residuals. SPM estimates the temporal autocorrelation
% from the residuals and stores the whitening matrix (square root of the
% inverse autocorrelation) in SPM.xX.W. For first-level models estimated
% with the "classical" ReML approach, W is typically diagonal or banded.
W = SPM.xX.W;

if issparse(W)
    W = full(W);
end

if size(W, 1) ~= n_scans || size(W, 2) ~= n_scans
    error('create_nomask_job:sizeMismatch', ...
          'W is [%d x %d] but expected [%d x %d].', ...
          size(W, 1), size(W, 2), n_scans, n_scans);
end

% -------------------------------------------------------------------------
% Extract the contrast vector
% -------------------------------------------------------------------------
% Our contrast of interest is con_0001, which is [1 0 0 ... 0] (the first
% regressor = "Stim", the effect of the pain stimulation condition). SPM
% stores contrast definitions in SPM.xCon, a struct array. We read the
% first one.
c = SPM.xCon(1).c(:);  % force column [n_reg x 1]

if length(c) ~= n_regressors
    error('create_nomask_job:sizeMismatch', ...
          'Contrast vector has %d elements but design has %d regressors.', ...
          length(c), n_regressors);
end

% -------------------------------------------------------------------------
% Identify the functional data file
% -------------------------------------------------------------------------
% SPM.xY.VY contains spm_vol headers for every scan. For a 4D NIfTI, all
% entries point to the same file with different volume indices. We extract
% the filename from the first entry.
%
% VY.fname may have a trailing ",1" (volume index) — strip it.
func_file = SPM.xY.VY(1).fname;
func_file = regexprep(func_file, ',\d+$', '');  % remove trailing ,N

if ~exist(func_file, 'file')
    warning('create_nomask_job:funcNotFound', ...
            'Functional file not found: %s\nIt may be on a different filesystem.', ...
            func_file);
end

% -------------------------------------------------------------------------
% Assemble the job structure
% -------------------------------------------------------------------------
job.spm_mat_path = spm_mat_path;
job.output_dir   = output_dir;
job.output_file  = fullfile(output_dir, 'con_0001.nii');
job.pKX          = pKX;
job.gSF          = gSF;
job.X0           = X0;
job.X0_pinv      = X0_pinv;
job.W            = W;
job.c            = c;
job.func_file    = func_file;
job.n_scans      = n_scans;
job.n_regressors = n_regressors;

end
