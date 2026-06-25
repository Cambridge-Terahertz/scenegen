function w = compute_density_weights_gpu(TARG, voxel_size,eps)
% Fast voxel-based density weighting (GPU-friendly)
% TARG: Nx3 scatterer locations
% voxel_size: scalar voxel pitch (meters)
    if nargin< 3
        eps = .1;
    end
    useGPU = isa(TARG,'gpuArray');
    if useGPU
        T = gather(TARG);   % indexing ops easier on CPU
    else
        T = TARG;
    end

    % Bounds
    xmin = min(T(:,1)); xmax = max(T(:,1));
    ymin = min(T(:,2)); ymax = max(T(:,2));
    zmin = min(T(:,3)); zmax = max(T(:,3));

    % Expand bounds slightly to avoid edge NaNs
    pad = voxel_size * 1e-3;
    xmin = xmin - pad; xmax = xmax + pad;
    ymin = ymin - pad; ymax = ymax + pad;
    zmin = zmin - pad; zmax = zmax + pad;

    % Voxel edges
    xedges = xmin:voxel_size:xmax;
    yedges = ymin:voxel_size:ymax;
    zedges = zmin:voxel_size:zmax;

    nx = numel(xedges)-1;
    ny = numel(yedges)-1;
    nz = numel(zedges)-1;

    % Map points to voxel indices
    ix = discretize(T(:,1), xedges);
    iy = discretize(T(:,2), yedges);
    iz = discretize(T(:,3), zedges);

    % Fix NaNs (outside edges)
    ix(isnan(ix)) = 1;
    iy(isnan(iy)) = 1;
    iz(isnan(iz)) = 1;

    % Clamp to valid ranges
    ix = max(1, min(nx, ix));
    iy = max(1, min(ny, iy));
    iz = max(1, min(nz, iz));

    % Linear voxel index
    lin_idx = sub2ind([nx, ny, nz], ix, iy, iz);

    % Count points per voxel
    counts = accumarray(lin_idx, 1, [nx*ny*nz, 1]);

    % Lookup local density
    local_count = counts(lin_idx);

    % Inverse-density weight
    w = 1 ./ (local_count + eps);

    % Normalize
    w = w ./ max(w);

    if useGPU
        w = gpuArray(single(w));
    else
        w = single(w);
    end
end
