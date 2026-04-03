function out = chckdelslurm(matoutfiles,shfiles,logfiles,matinfiles,folder,delout,verbose)

if nargin==6
    verbose = false;
end

e = false;
sent = false;
while ~e
    [e,sent] = esxistfiles(matoutfiles,verbose,sent);
end

N = numel(matoutfiles);
out = cell(N,1);
for i = 1:N
    delete(shfiles{i});
    delete(logfiles{i});
    delete(matinfiles{i});
    out{i} = load(matoutfiles{i});
    if delout
        delete(matoutfiles{i});
    end
end

if isempty(folder)
    folder = pwd;
end
outfiles = dir([folder '/slurm-*.out']);
for i = 1:numel(outfiles)
    delete(fullfile(folder,outfiles(i).name));
end
outfiles = dir([pwd '/slurm-*.out']);
for i = 1:numel(outfiles)
    delete(fullfile(pwd,outfiles(i).name));
end
corefiles = dir([folder '/core.*.ufhpc*']);
for i = 1:numel(corefiles)
    delete(fullfile(folder,corefiles(i).name));
end

function [e,sent] = esxistfiles(files,verbose,sent)

N = numel(files);
e = false(N,1);
for i = 1:N
    e(i) = exist(files{i},'file');
end
if verbose&&~all(e)
    disp('Press a key to learn which tasks are missing...')
    disp(files(~e))
    if sent
        pause
    else
        result = input('These tasks were not sent. Do you want to resend them? (y/n)\n','s');
        if strcmp(result,'y')
            rerunshfiles
            sent = true;
        end
    end
end
e = all(e);