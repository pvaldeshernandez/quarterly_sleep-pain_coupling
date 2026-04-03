function cellnames = reslice_spacen(I0,vx,order)

for image = 1:size(I0,1)
    I = I0(image,:);
    V = spm_vol(I); n = length(V);
    vx0 = sqrt(sum(V(1).mat(1:3,1:3).^2));
    Vf = V(1); Vf.mat = V(1).mat/(diag([vx0 1]))*diag([vx 1]);
    a = diag([vx 1])\diag([vx0 1])*[V(1).dim(1:3) 1]';
    Vf.dim(1:3) = round(abs(a(1:3))');

    [a1,a2,a3] = fileparts(I);
    counter = 0; %handle = waitbar(counter,'resampling...');
    [X,Y] = ndgrid(1:Vf.dim(1),1:Vf.dim(2));
    names = '';
    for i = 1:n,
        Vn = Vf; c = spm_bsplinc(V(i),[order*ones(1,3) 0 0 0]);
        Vn.pinfo = V(i).pinfo; Vn.dt = V(i).dt; Vn.n = V(i).n;
        [pathstr,name,ext] = fileparts(V(i).fname);
        Vn.fname = fullfile(pathstr,['r' name ext]);
        names = strvcat(names,Vn.fname);
        Vn = rmfield(Vn,'private');
        Vn = spm_create_vol(Vn); M = V(i).mat\Vn.mat;
        for z = 1:Vn.dim(3),
            A = spm_bsplins(c,M(1,1)*X + M(1,2)*Y + M(1,3)*z + M(1,4),...
                M(2,1)*X + M(2,2)*Y + M(2,3)*z + M(2,4),...
                M(3,1)*X + M(3,2)*Y + M(3,3)*z + M(3,4),...
                [order*ones(1,3) 0 0 0]);
            spm_write_plane(Vn,A,z);  %waitbar(counter/(Vn.dim(3)*n),handle)
            counter = counter+1;
        end,
    end, %close(handle);
    cellnames{image} = unique(deblank(names),'rows');
end
fclose all;