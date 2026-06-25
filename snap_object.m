% ------------------------------------------------------------
% Load point clouds
% ------------------------------------------------------------
body  = read_xyz_binary('D:\model_exports\f25.1_001\xyz\frame_0001.xyz');     % N×3
asset = read_xyz_binary('D:\model_exports\f25.1_001\xyz\gun.xyz');    % M×3

% ------------------------------------------------------------
% Compute waist height (50% of body height)
% ------------------------------------------------------------
z = body(:,3);
z_min = min(z);
z_max = max(z);

waist_z = z_min + 0.50 * (z_max - z_min);

% Use a small band around the waist plane
band = abs(z - waist_z) < 0.01;   % 1 cm band
waist_point = mean(body(band, :), 1);   % 1×3 centroid

% ------------------------------------------------------------
% Translate asset so its center sits at the waist
% ------------------------------------------------------------
asset_center = mean(asset, 1);
asset_translated = asset - asset_center + waist_point;

% ------------------------------------------------------------
% Visualization
% ------------------------------------------------------------
figure; hold on;
scatter3(body(:,1), body(:,2), body(:,3), 1, 'b');
scatter3(asset_translated(:,1), asset_translated(:,2), asset_translated(:,3), 5, 'r');
axis equal
xlabel X; ylabel Y; zlabel Z;
title('Asset placed at waist');
