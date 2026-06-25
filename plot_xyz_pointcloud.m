function plot_xyz_pointcloud(xyz_file, varargin)
% plot_xyz_pointcloud Load and visualize a binary float32 XYZ point cloud.
%
%   plot_xyz_pointcloud('model.xyz')
%
%   Optional arguments:
%       'Color', [r g b]   → specify point color (0–1)
%       'MarkerSize', s    → point size (default 20)
%
%   The .xyz file must contain float32 triplets: x y z x y z ...

    % ------------------------------------------------------------
    % Parse inputs
    % ------------------------------------------------------------
    p = inputParser;
    addRequired(p, 'xyz_file', @ischar);
    addParameter(p, 'Color', [0.2 0.6 1.0], @(x) isnumeric(x) && numel(x)==3);
    addParameter(p, 'MarkerSize', 20, @isscalar);
    parse(p, xyz_file, varargin{:});

    color = p.Results.Color;
    msize = p.Results.MarkerSize;

    % ------------------------------------------------------------
    % Read binary XYZ (float32)
    % ------------------------------------------------------------
    fid = fopen(xyz_file, 'rb');
    if fid < 0
        error('Could not open file: %s', xyz_file);
    end

    data = fread(fid, 'float32');
    fclose(fid);

    % Reshape into Nx3
    if mod(numel(data), 3) ~= 0
        error('File does not contain a valid XYZ float32 triplet structure.');
    end

    pts = reshape(data, 3, []).';
    
    % ------------------------------------------------------------
    % Convert to MATLAB pointCloud object
    % ------------------------------------------------------------
    pc = pointCloud(pts);

    % ------------------------------------------------------------
    % Plot using pcshow
    % ------------------------------------------------------------
    figure('Color', 'w');
    pcshow(pc, 'MarkerSize', msize);
    title(sprintf('Point Cloud: %s', xyz_file), 'Interpreter', 'none');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;

    % Apply color after pcshow (MATLAB quirk)
    hold on;
    scatter3(pts(:,1), pts(:,2), pts(:,3), msize, color, 'filled');
    hold off;
end
