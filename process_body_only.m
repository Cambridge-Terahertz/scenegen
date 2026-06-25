function [TARG, S] = process_body_only(body_file, body_s)
% PROCESS_BODY_ONLY
% Reproduces the exact body-processing pipeline from place_assets.m
% but without loading or attaching any assets.

fprintf('[%s] Loading BODY ONLY: %s\n', datetime("now"), body_file);

% ------------------------------------------------------------
% Load body (same loader used in place_assets)
% ------------------------------------------------------------
body = read_xyz_binary(body_file);

% ------------------------------------------------------------
% Apply same axis permutation as place_assets
% body = body(:,[3 1 2]);
% ------------------------------------------------------------
body = body(:, [3 1 2]);

% ------------------------------------------------------------
% Compute Z-centering BEFORE rotation (same as place_assets)
% ------------------------------------------------------------
mn   = mean(body);
mx   = max(body(:,3));
mns  = min(body(:,3));
center = (mx + mns) / 2;
mn(3)  = center;

body = body - mn;

% ------------------------------------------------------------
% Apply same -90° rotation around X as place_assets
% ------------------------------------------------------------
TM_b = transform(-90*pi/180, 0, 0);
body_h = [body ones(size(body,1),1)];
body_rot = TM_b * body_h.';
body = body_rot(1:3,:).';

% ------------------------------------------------------------
% Output format matches place_assets
% ------------------------------------------------------------
TARG = body;
S.s = body_s .* ones(size(body,1),1);

fprintf('[%s] Body-only processing complete. %d points.\n', ...
    datetime("now"), size(TARG,1));

end

% ------------------------------------------------------------
% Helper: identical to place_assets
% ------------------------------------------------------------
function pts = read_xyz_binary(filename)
fprintf('[%s] Loading %s...\n', datetime("now"), filename);
fid = fopen(filename, 'rb');
data = fread(fid, inf, 'float32=>single');
fclose(fid);
pts = reshape(data, 3, []).';
end
