function [img, info] = bp_freq_(p)
% bp_freq_  Tiled, GPU frequency-domain backprojection with phase-center (monostatic) geometry
%
% p fields:
%   f1, f2, nf
%   sig          [nf, nr_y, nr_x] or [nf, nXcvr]
%   tx           [nTx x 3] (assumes 1 TX)
%   rx           [nXcvr x 3]
%   nr_elements
%   spatial_window: 'none' or 'tukeywin'
%   freq_window   : 'none' or 'hann'
%   xx_recon, yy_recon, zz_recon (3D grids)
%   nx_recon, ny_recon, nz_recon

t_total = tic;

%% Constants and frequency axis
c  = 299792458;
f  = linspace(p.f1, p.f2, p.nf).';   % [nf x 1]

c  = gpuArray(single(c));
f  = gpuArray(single(f));
tx = gpuArray(single(p.tx));
rx = gpuArray(single(p.rx));

%% Assume single TX; if not, you can extend this later
tx0 = tx(1,:);   % [1 x 3]

%% Spatial window
if isfield(p, "spatial_window") && strcmpi(p.spatial_window, "tukeywin")
    winx  = tukeywin(p.nr_elements, 0.95);
    winy  = tukeywin(p.nr_elements, 0.95);
    winxy = winx * winy.';
    for k = 1:p.nf
        p.sig(k,:,:) = squeeze(p.sig(k,:,:)) .* winxy;
    end
end

%% Frequency window
winf = 1;
if isfield(p, "freq_window") && strcmpi(p.freq_window, "hann")
    winf = hann(p.nf);
end

%% Move data to GPU
S_f = gpuArray(single(reshape(p.sig, p.nf, [])));   % [nf x nXcvr]
winf = gpuArray(single(winf(:)));
S_f  = S_f .* winf;                                 % apply freq window

%% Recon grid (flattened)
xx = gpuArray(single(p.xx_recon(:)));
yy = gpuArray(single(p.yy_recon(:)));
zz = gpuArray(single(p.zz_recon(:)));

nVox  = numel(xx);
nXcvr = size(S_f, 2);
nf    = p.nf;

%% Init image
img = gpuArray.zeros(nVox,1,'single');

fprintf('[%s] freq-domain recon (phase-center, tiled) starting...\n', datetime("now"))
t_bp = tic;

%% Tiling parameters (tune if needed)
VOX_CHUNK = 350e4;   % ~50k voxels per tile
nChunks   = ceil(nVox / VOX_CHUNK);

for ixcvr = 1:nXcvr

    % receiver position
    rx_i = rx(ixcvr,:);   % [1 x 3]

    % phase center for this TX–RX pair
    pc_i = 0.5 * (tx0 + rx_i);   % [1 x 3]

    % data for this receiver: [nf x 1]
    Sf_i = S_f(:, ixcvr);        % [nf x 1]

    for ich = 1:nChunks
        idx_start = (ich-1)*VOX_CHUNK + 1;
        idx_end   = min(ich*VOX_CHUNK, nVox);
        idx       = idx_start:idx_end;

        % voxel subset
        xx_c = xx(idx);
        yy_c = yy(idx);
        zz_c = zz(idx);

        % distance from phase center to voxel
        dPC_c = sqrt( (pc_i(1)-xx_c).^2 + ...
                      (pc_i(2)-yy_c).^2 + ...
                      (pc_i(3)-zz_c).^2 );   % [Nv_c x 1]

        % monostatic equivalent total path: R = 2 * dPC
        T_c = (2 .* dPC_c) ./ c;             % [Nv_c x 1]

        % phase matrix: [Nv_c x nf]
        phase = exp(1j * 2 * pi * (T_c * f.'));   % [Nv_c x nf]

        % accumulate: img_c += phase * Sf_i
        img(idx) = img(idx) + phase * Sf_i;       % [Nv_c x 1]
    end
end

t_bp = toc(t_bp);
fprintf('[%s] freq-domain recon complete [%0.2f sec]\n', datetime("now"), t_bp)

%% Reshape and gather
img = gather(img);
img = reshape(img, p.nx_recon, p.ny_recon, p.nz_recon);

info.bp_recon_timing      = t_bp;
info.total_process_timing = toc(t_total);
info.nr_voxels            = nVox;
info.array_elements       = nXcvr;
info.nr_tx                = size(tx,1);
end
