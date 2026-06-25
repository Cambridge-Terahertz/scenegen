function [img, info] = bp_(p)
% bp_  Time-domain backprojection image reconstruction (GPU)
%
% Required fields in p:
%   f1, f2           : start/end frequency (Hz)
%   nf               : number of frequency samples
%   npad             : IFFT length (time samples)
%   sig              : measured data, size [nf, nr_y, nr_x] or [nf, nXcvr]
%   tx               : transmitter position(s), [nTx x 3] (assumes 1 TX)
%   rx               : receiver positions, [nXcvr x 3]
%   nr_elements      : array elements per side (for spatial window)
%   spatial_window   : 'none' or 'tukeywin'
%   freq_window      : 'none' or 'hann'
%   xx_recon, yy_recon, zz_recon : recon grid (3D arrays)
%   nx_recon, ny_recon, nz_recon : recon grid sizes

t_total = tic;

%% Constants and frequency axis
c  = 299792458;
f  = linspace(p.f1, p.f2, p.nf);   % [1 x nf]

c  = gpuArray(single(c));
f  = gpuArray(single(f));
tx = gpuArray(single(p.tx));
rx = gpuArray(single(p.rx));

%% Spatial window
if isfield(p, "spatial_window") && strcmpi(p.spatial_window, "tukeywin")
    winx  = tukeywin(p.nr_elements, 0.95);
    winy  = tukeywin(p.nr_elements, 0.95);
    winxy = winx * winy.';   % [nr_elements x nr_elements]

    % p.sig assumed [nf, nr_y, nr_x]
    for ifreq = 1:p.nf
        p.sig(ifreq,:,:) = squeeze(p.sig(ifreq,:,:)) .* winxy;
    end
end

%% Frequency window
winf = 1;
if isfield(p, "freq_window") && strcmpi(p.freq_window, "hann")
    winf = hann(p.nf);   % [nf x 1]
end

%% Reshape signal and move to GPU
p.sig = reshape(p.sig, p.nf, []);          % [nf x nXcvr]
sig   = gpuArray(single(p.sig));
winf  = gpuArray(single(winf(:)));         % [nf x 1]

%% IFFT to time domain (correct stepped-frequency scaling)
df = f(2) - f(1);              % frequency spacing
dt = 1 / (df * p.npad);        % time resolution for IFFT length npad
ti = (0:p.npad-1) * dt;        % [1 x npad], CPU

td_ = ifft(sig .* winf, p.npad, 1);   % [npad x nXcvr], GPU

%% Recon grid (flattened)
xx = gpuArray(single(p.xx_recon(:)));
yy = gpuArray(single(p.yy_recon(:)));
zz = gpuArray(single(p.zz_recon(:)));

nVox  = numel(xx);
nXcvr = size(td_, 2);

%% Geometry-derived Tmin/Tmax with safety margin
X = [min(xx) max(xx)];
Y = [min(yy) max(yy)];
Z = [min(zz) max(zz)];
[Xc, Yc, Zc] = ndgrid(X, Y, Z);
corners = [Xc(:) Yc(:) Zc(:)];   % 8 x 3

Tmin = inf;
Tmax = -inf;

% assume single TX (tx(1,:)); extend later if you encode TX/RX pairing
tx0 = tx(1,:);

for k = 1:size(corners,1)
    r   = corners(k,:);                 % [1 x 3]
    dTV = vecnorm(tx0 - r, 2, 2);       % scalar
    dVR = vecnorm(rx  - r, 2, 2);       % [nXcvr x 1]
    T   = (dTV + dVR) ./ c;             % [nXcvr x 1]

    Tmin = min(Tmin, min(T));
    Tmax = max(Tmax, max(T));
end

% safety margin so we don't clip depth
Tspan   = Tmax - Tmin;
margin  = 0.5 * Tspan;                  % 50% extra on each side
Tmin_m  = max(0, Tmin - margin);
Tmax_m  = Tmax + margin;
Tmid    = 0.5 * (Tmin_m + Tmax_m);

% center time axis on expanded mid (no rescaling, just shift)
ti = ti - mean(ti) + Tmid;              % [1 x npad], CPU
ti = gpuArray(single(ti));              % [1 x npad], GPU

%% Backprojection
fprintf('[%s] recon starting...\n', datetime("now"))

p0  = gpuArray(single(1j * 2 * pi * p.f1));  % phase factor at f1
img = gpuArray.zeros(nVox, 1, 'single');

t_bp = tic;
for ixcvr = 1:nXcvr
    % distance TX -> voxel (keep your original sign convention)
    dTV = sqrt( (-tx0(1) - xx).^2 + ...
                (-tx0(2) - yy).^2 + ...
                ( tx0(3) - zz).^2 );

    % distance voxel -> RX(ixcvr,:)
    dVR = sqrt( (rx(ixcvr,1) - xx).^2 + ...
                (rx(ixcvr,2) - yy).^2 + ...
                (rx(ixcvr,3) - zz).^2 );

    % total travel time
    T = (dTV + dVR) ./ c;   % [nVox x 1]

    % phase correction at f1
    phCorr = exp(p0 .* T);

    % interpolate time-domain data at T
    s_t = interp1(ti, td_(:, ixcvr), T, "linear", 0);  % [nVox x 1]

    % accumulate image
    img = img + s_t .* conj(phCorr);
end
t_bp = toc(t_bp);

fprintf('[%s] recon complete [%0.2f sec]\n', datetime("now"), t_bp)

%% Reshape and gather
img  = gather(img);
img  = reshape(img, p.nx_recon, p.ny_recon, p.nz_recon);

info.bp_recon_timing      = t_bp;
info.total_process_timing = toc(t_total);
info.nr_voxels            = nVox;
info.array_elements       = size(rx, 1);
info.nr_tx                = size(tx, 1);
end
