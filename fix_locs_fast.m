function [data_out, keep_inds, map] = fix_locs_fast(data, r0)
% Enforce minimum spacing r0 using voxel hashing (O(N))
% Returns:
%   data_out  = filtered point cloud
%   keep_inds = logical mask of kept points (Nx1)
%   map       = indices of kept points (useful for mapping other arrays)

if r0 == 0
    data_out  = data;
    keep_inds = true(size(data,1),1);
    map       = (1:size(data,1))';
    return
end

N = size(data,1);

% --- 1. Compute voxel indices ---
voxel = r0;   % voxel size = minimum spacing

xmin = min(data(:,1)); 
ymin = min(data(:,2)); 
zmin = min(data(:,3));

ix = floor((data(:,1) - xmin) / voxel);
iy = floor((data(:,2) - ymin) / voxel);
iz = floor((data(:,3) - zmin) / voxel);

% --- 2. Hash voxel triplets into a single integer ---
% Large primes reduce collisions
hash = ix + 73856093*iy + 19349663*iz;

% --- 3. Keep only the first point in each voxel ---
[~, map] = unique(hash, 'stable');

% map = indices of points to keep
keep_inds = false(N,1);
keep_inds(map) = true;

% Filtered point cloud
data_out = data(map,:);
