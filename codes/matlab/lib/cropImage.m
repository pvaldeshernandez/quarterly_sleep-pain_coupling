function V = cropImage(I,borders,th,option)

%CROPIMAGE crops an image to remove empty voxels in its borders.
%
% Usage: V = CROPIMAGE(FILENAME,BORDERS,THRESHOLD,RW_OPTION)
%
% FILENAME: image filename string including its pathname.
%
% BORDERS: a vector specifying the number of voxels without brain tissue that
% will be left in the borders of the image at each side. It has to be of length
% 3 to set each dimension of the image.
%
% THRESHOLD: voxels with intensity values below THRESHOLD are considered
% empty and therefore eliminated unless they are in the range specified by
% BORDERS.
%
% RW_OPTION: 0 if original image is overwritten by cropped image, 1
% otherwise.
%
% V: SPM volume of the cropped image.
%
% An image file will be created in the same folder of FILENAMES(1) with
% prefix 'c'.
%
% cropping image can reduce make computation time, and memory consumption.
%
% See also: EXTRACT_BRAIN, MAKEMASKS
%
% Author: Pedro Antonio Valdes-Hernandez
% Neuroimaging Department
% Cuban Neuroscience Center
% November 18 2005
% Version $1.7

V = spm_vol(I);
P = spm_read_vols(V);
S = size(P);
borders = 2*borders;

ind = find(P > th);
[x,y,z] = ind2sub(S,ind);

P = P(min(x):max(x),min(y):max(y),min(z):max(z));
S = size(P);
P(end + borders(1),end + borders(2),end + borders(3)) = 0;
d = round((size(P)/2) - (S/2));
P = translateImageN0(P,d(1),d(2),d(3));

[pathstr,name,ext] = fileparts(V.fname);
if option || nargin == 3
    V.fname = fullfile(pathstr,['c' name ext]);
end
V.dim(1:3) = size(P);
vec = [min(x)-1 min(y)-1 min(z)-1] - borders/2; vec = V.mat(1:3,1:3)*vec';
V.mat(1:3,4) = V.mat(1:3,4) + vec;
V = rmfield(V,'private');

spm_write_vol(V,P);

V = spm_vol(V.fname);