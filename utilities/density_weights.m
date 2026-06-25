function w = density_weights(X, h)
% X is Nx3 scatterer locations
% h is kernel bandwidth (meters)

N = size(X,1);
tree = KDTreeSearcher(X);

% radius for local neighborhood (3 sigma)
r = 3*h;

% Preallocate
rho = zeros(N,1);

for i = 1:N
    idx = rangesearch(tree, X(i,:), r);
    idx = idx{1};

    % Gaussian kernel density
    d2 = sum((X(idx,:) - X(i,:)).^2, 2);
    rho(i) = sum(exp(-d2/(2*h^2)));
end

% Weight = inverse density
w = 1 ./ (rho + eps);

% Normalize
w = w ./ max(w);
