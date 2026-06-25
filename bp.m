function [img,info] = bp(p)
% time domian back projection image reconstruction
% note, this recon will work with an arbitrary arrangement of transmitters
% and receivers and can reconstruct an arbitraty volume (point cloud).
% Recon space does not need to be on a regular grid.


% inputs fields for p

% xx_recon: recon locations, cross range
% yy_recon: recon locations, height
% zz_recon: recon locations, depth

% npad: padding amount for recon typically 5 x nf
% spatial_window: (true|false), apply spatial window to reduce cross range 
%                 sidelobes: hardcoded ==> tukeywin(nr_elements,.95)
% freq_window: (true|false), apply frequency window to reduce down range
%              sidelobes: hardcoded ==> hann(nf)

% f1: start freqeuncy (275e9Hz)
% f2: end frequency   (305e9Hz)
% nf: # of frequency measurements (256)

% tx: transimtter locations [m x 3] [xpos ypos zpos]
% rx: receiver locations [nr_elements^2 x 3]
% nr_elements: assumes array is square, # of elements along one side 
t3 = tic;
% fprintf('[%s] resetting gpu...\n',datetime("now"))
% gpuDevice(1);

fprintf('[%s] prepping data for recon...\n',datetime("now"))
 c = 299792458;
 f = linspace(p.f1,p.f2,p.nf);


c = gpuArray(single(c));
tx = gpuArray(single(p.tx));
rx = gpuArray(single(p.rx));
f = gpuArray(single(f));

% sig_ = reshape(p.sig,p.nr_elements,p.nr_elements,p.nf);
winxy = 1;
winf = 1;
switch p.spatial_window
    case 'tukeywin'
        winx = tukeywin(p.nr_elements,.95);
        winy = tukeywin(p.nr_elements,.95);
        winxy = winx*winy';

        for ifreq = 1:p.nf
            p.sig(ifreq,:,:) = squeeze(p.sig(ifreq,:,:)).*winxy;
        end
end

switch p.freq_window
    case 'hann'
    winf=hann(p.nf);

    
end


p.sig = reshape(p.sig,p.nf,[]);

df = f(2)-f(1);
BW = df*p.npad;
dt = 1/BW;
ti = 0:dt:dt*(p.npad-1);
td_ = ifft(p.sig.*winf,p.npad);


% nx_recon = 61;
% ny_recon = 61;
% nz_recon = 251;

% xs = linspace(p.x1_recon,p.x2_recon,p.nx_recon);  % width
% ys = linspace(p.y1_recon,p.y2_recon,p.ny_recon);  % height
% zs = linspace(p.z1_recon,p.z2_recon,p.nz_recon);  % depth
% 
% [xx_recon,yy_recon,zz_recon] = ndgrid(xs,ys,zs);



xx_recon = gpuArray(single(p.xx_recon));
yy_recon = gpuArray(single(p.yy_recon));
zz_recon = gpuArray(single(p.zz_recon));


p0 = gpuArray(single(1j*2*pi*p.f1));
nxcvr = size(td_,2);
td_ = gpuArray(single(td_));
ti = gpuArray(single(ti));

%==========================================================================
% shift time vector so it overlaps with voxel times. Need to do this
% particularly when using only a small number of frequencies in the model.
%==========================================================================
z2 = max(zz_recon(:));
z1 = min(zz_recon(:));


x2 = max(xx_recon(:));
x1 = min(xx_recon(:));

y2 = max(yy_recon(:));
y1 = min(yy_recon(:));
r1 = sqrt(x1.^2+y1.^2+z1.^2);
r2 = sqrt(x2.^2+y2.^2+z2.^2);
% mid = (r2+r1)/2;
mid = (z2+z1)/2;
t_offset = 2*mid./c;
t_min = 2*z1/c;

if ti(end)< t_min
    % ti = (ti-mean(ti))+t_min;
     ti = (ti-mean(ti))+t_offset;
end


%==========================================================================

img = 0;
t1 = tic;
fprintf('[%s] recon starting...\n',datetime("now"))
for ixcvr=1:nxcvr % loop over transmitter receiver pairs

    % distance transmitter to voxel
    dTV = sqrt((-tx(1)-xx_recon).^2+(-tx(2)-yy_recon).^2+(tx(3)-zz_recon).^2);

    % distance voxel to receiver
    dVR = sqrt((rx(ixcvr,1)-xx_recon).^2+(rx(ixcvr,2)-yy_recon).^2+(rx(ixcvr,3)-zz_recon).^2);

    % Total travel time
    T = (dTV+dVR)./c;
    phCorr = exp(p0.*T);
    img = img+interp1(ti,squeeze(td_(:,ixcvr)),T,"linear",0).*conj(phCorr);

end
t2 = toc(t1);
fprintf('[%s] recon complete [%0.2fsec]\n',datetime("now"),t2)

img = gather(img);
img = reshape(img,p.nx_recon,p.ny_recon,p.nz_recon);
info.bp_recon_timing = t2;
info.total_process_timing= toc(t3);
info.nr_voxels = numel(xx_recon);
info.array_elements = size(rx,1);
info.nr_tx = size(tx,1);

