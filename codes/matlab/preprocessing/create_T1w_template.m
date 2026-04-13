% CREATE_T1W_TEMPLATE  Build the study-specific DARTEL template.
%
% =========================================================================
% PIPELINE POSITION: Step 2 (Group A — structural stream)
% =========================================================================
%
% PURPOSE
% -------
% Construct a study-specific anatomical template using DARTEL. Starting
% from the Procrustes-aligned tissue segmentations (rc1, rc2, rc3) of the
% session-01 T1-weighted images, DARTEL iteratively estimates diffeomorphic
% deformation fields that bring all subjects into a common space. This
% yields a series of increasingly sharp group templates (_0 through _6)
% and per-subject flow fields (u_rc1*) encoding each individual's
% deformation.
%
% The study-specific template provides superior inter-subject alignment
% compared to normalization to a generic atlas (e.g., MNI152), because it
% is derived from the actual sample and preserves population-specific
% anatomical features. This is especially important for subcortical ROIs
% (NAcc, thalamic nuclei) used in this study's moderation analyses.
%
% DARTEL PARAMETERS (matching manuscript)
% ----------------------------------------
%   Input tissues           : rc1 (GM), rc2 (WM), rc3 (CSF)
%   Number of outer iterations: 6
%   Inner iterations per outer: 3
%   Regularization form     : 0 (linear elastic)
%
%   Iteration parameters [rparam, K, slam]:
%     Iter 1: rparam=[4 2 1e-6],     K=0, slam=16
%     Iter 2: rparam=[2 1 1e-6],     K=0, slam=8
%     Iter 3: rparam=[1 0.5 1e-6],   K=1, slam=4
%     Iter 4: rparam=[0.5 0.25 1e-6], K=2, slam=2
%     Iter 5: rparam=[0.25 0.125 1e-6], K=4, slam=1
%     Iter 6: rparam=[0.25 0.125 1e-6], K=6, slam=0.5
%
%   rparam = [a1 a2 a3] controls the balance between membrane energy (a1),
%   bending energy (a2), and linear-elastic energy (a3). Decreasing values
%   across iterations allow progressively finer deformations.
%
%   K = number of time steps for the diffeomorphic integration (2^K).
%   Increasing K allows larger deformations at later iterations.
%
%   slam = scaling of the time step (Levenberg-Marquardt-like
%   regularization). Decreasing values yield larger updates.
%
%   Optimization:
%     lmreg = 0.01 (Levenberg-Marquardt regularization)
%     cyc   = 3    (number of multigrid cycles)
%     its   = 3    (number of relaxation iterations per cycle)
%
% EXPECTED INPUTS
% ---------------
%   Procrustes-aligned segmentations from session 01:
%     <folder>/ses-01/anat/rc1sub-*_ses-01*_T1w.nii  (GM)
%     <folder>/ses-01/anat/rc2sub-*_ses-01*_T1w.nii  (WM)
%     <folder>/ses-01/anat/rc3sub-*_ses-01*_T1w.nii  (CSF)
%
% OUTPUTS
% -------
%   Written to <folder>/ses-01/anat/:
%     <study>_T1w__0.nii ... <study>_T1w__6.nii  — Template iterations
%     u_rc1sub-*_ses-01*_T1w.nii                  — Per-subject flow fields
%
% REFERENCES
% ----------
%   Ashburner J. A fast diffeomorphic image registration algorithm.
%   NeuroImage. 2007;38(1):95-113.
%
% VARIABLES FROM script_study.m
% -----------------------------
%   folder, study, account, burst, mem, time, user, matlabver, slurmfolder
%
% NOTE
% ----
%   This is computationally expensive (32 GB RAM, up to 72 hours). The
%   burst QOS is disabled and a dedicated SLURM configuration is used.
%
% =========================================================================

% Specify Procrustes-aligned segmentations.
rc1s = filedir2cell(dir([folder 'ses-01/anat/rc1sub-*_ses-01*_T1w.nii']));
rc2s = filedir2cell(dir([folder 'ses-01/anat/rc2sub-*_ses-01*_T1w.nii']));
rc3s = filedir2cell(dir([folder 'ses-01/anat/rc3sub-*_ses-01*_T1w.nii']));

% Create batch.
clear matlabbatch
matlabbatch{1}.spm.tools.dartel.warp.images = {rc1s, rc2s, rc3s};
matlabbatch{1}.spm.tools.dartel.warp.settings.template = [study '_T1w_'];
matlabbatch{1}.spm.tools.dartel.warp.settings.rform = 0;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).rparam = [4 2 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).K = 0;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 16;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).rparam = [2 1 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).K = 0;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).slam = 8;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).rparam = [1 0.5 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).K = 1;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).slam = 4;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).rparam = [0.5 0.25 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).K = 2;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).slam = 2;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).rparam = [0.25 0.125 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).K = 4;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).slam = 1;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).its = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).rparam = [0.25 0.125 1e-06];
matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).K = 6;
matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).slam = 0.5;
matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.01;
matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc = 3;
matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its = 3;

% Send the batches to SLURM.
strrun = 'run';
conf = struct('account',account,'burst',0,'mem',32,'time','72:00:00','user',user,'matlabver',matlabver);
clear matoutfiles shfiles logfiles matinfiles
[matoutfiles{1},shfiles{1},logfiles{1},matinfiles{1}] = sendslurm('','spm_jobman',slurmfolder,conf,strrun,matlabbatch);
% Clean the files needed to run the batches in the SLURM.
chckdelslurm(matoutfiles,shfiles,logfiles,matinfiles,slurmfolder,1);