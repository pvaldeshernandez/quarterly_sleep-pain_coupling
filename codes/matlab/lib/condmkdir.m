function condmkdir(folder)

if ~exist(folder,'file')
    mkdir(folder)
end