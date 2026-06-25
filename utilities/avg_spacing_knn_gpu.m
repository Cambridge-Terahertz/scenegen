function d_avg = avg_spacing_knn_gpu(T, k)
% T: Nx3 gpuArray or CPU array
% k: nearest neighbor index

useGPU = isa(T,'gpuArray');

if useGPU
    Tcpu = gather(T);
else
    Tcpu = T;
end

Mdl = KDTreeSearcher(Tcpu);
[idx, dist] = knnsearch(Mdl, Tcpu, 'K', k+1);

d = dist(:, end);   % k-th neighbor

d_avg = median(d);
