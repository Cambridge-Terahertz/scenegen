function [TARG, S] = place_assets(b, a, doQC)
addpath(genpath('D:\git_sandbox\model_gen\utilities'))

arots = a.prerotations;
atrans = a.preplacement;

tilt_angle_deg    = a.final_tilt;
extra_translation = a.final_trans;

% ============================================================
% Load and transform BODY
% ============================================================
body  = read_xyz_binary(b.files);

body = body(:,[3 1 2]);
TM_b = transform(-90*pi/180,0,0);

mn   = mean(body);
mx   = max(body(:,3));
mns  = min(body(:,3));
center = (mx+mns)/2;
mn(3)  = center;

body = body - mn;
body = [body ones(size(body,1),1)];
body = TM_b*body.';
body = body(1:3,:).';

% ============================================================
% BODY METADATA (post-transform)
% ============================================================
body_bbox_min = min(body,[],1);
body_bbox_max = max(body,[],1);
body_center   = mean(body,1);

S = struct();
S.body.file      = b.files;
S.body.scale     = b.s;
S.body.bbox_min  = body_bbox_min;
S.body.bbox_max  = body_bbox_max;
S.body.center    = body_center;

% ============================================================
% PROCESS ASSETS
% ============================================================
all_assets = [];
s_assets   = [];
nassets    = numel(a.files);

asset_meta = struct([]);

% PCA of body for relative rotation reference
[coeff_body,~] = pca(body - body_center);
R_body = coeff_body;   % 3×3

for iasset = 1:nassets

    % -----------------------------
    % Load asset
    % -----------------------------
    asset_file = a.files{iasset};
    asset      = read_xyz_binary(asset_file);

    [~,base,~] = fileparts(asset_file);
    semantic_label = base;

    xt = arots(iasset,1);
    yt = arots(iasset,2);
    zt = arots(iasset,3);

    asset = asset - mean(asset);
    TM_a  = transform(xt,yt,zt);
    asset = [asset ones(size(asset,1),1)];
    asset = TM_a*asset.';
    asset = asset(1:3,:).';

    % -----------------------------
    % Pre-place asset (random jitter)
    % -----------------------------
    p = [0.1*rand, 0, 0.1*rand];
    preplacement = atrans(iasset,:) + p;
    asset = asset + preplacement;

    % -----------------------------
    % 1. Attachment point
    % -----------------------------
    asset_attach = mean(asset,1);

    % -----------------------------
    % 2. Closest body point
    % -----------------------------
    dists = vecnorm(body - asset_attach, 2, 2);
    [~, idx_min] = min(dists);
    body_hit = body(idx_min, :);

    % -----------------------------
    % 3. Local surface normal
    % -----------------------------
    d_local = vecnorm(body - body_hit, 2, 2);
    local   = body(d_local < 0.06, :);

    [coeff_local, ~] = pca(local);
    normal = coeff_local(:,3)';

    if dot(asset_attach - body_hit, normal) < 0
        normal = -normal;
    end
    normal = normal / norm(normal);

    % -----------------------------
    % 4. Thickness offset
    % -----------------------------
    proj      = asset * normal';
    thickness = max(proj) - min(proj);
    offset    = 0.25 * thickness + 0.003;

    target_point    = body_hit + offset * normal;
    translation_vec = target_point - asset_attach;

    asset_moved = asset + translation_vec;

    % -----------------------------
    % 5. Final tilt
    % -----------------------------
    [coeff_asset, ~] = pca(asset_moved - mean(asset_moved,1));
    local_axis = coeff_asset(:,1)';

    tilt_angle = deg2rad(tilt_angle_deg(iasset) + .3*randn(1));
    tilt_axis  = local_axis / norm(local_axis);
    R_tilt     = axang2rotm([tilt_axis tilt_angle]);

    asset_final = (R_tilt * (asset_moved - asset_attach)')' + asset_attach;
    asset_final = asset_final + extra_translation(iasset,:)+p;

    % ============================================================
    % RELATIVE TRANSFORMS (asset → body frame)
    % ============================================================
    asset_center = mean(asset_final,1);

    rel_translation = asset_center - body_center;

    [coeff_asset2,~] = pca(asset_final - asset_center);
    R_asset = coeff_asset2;

    R_rel = R_body' * R_asset;
    axang_rel = rotm2axang(R_rel);

    % ============================================================
    % STORE PER‑ASSET METADATA
    % ============================================================
    asset_meta(iasset).name                  = asset_file;
    asset_meta(iasset).file                  = asset_file;
    asset_meta(iasset).semantic_label        = semantic_label;

    asset_meta(iasset).translation           = translation_vec + extra_translation(iasset,:);
    asset_meta(iasset).rotation              = [tilt_axis tilt_angle];

    asset_meta(iasset).bbox_min              = min(asset_final,[],1);
    asset_meta(iasset).bbox_max              = max(asset_final,[],1);

    asset_meta(iasset).attach_body           = body_hit;
    asset_meta(iasset).normal                = normal;

    asset_meta(iasset).relative_translation  = rel_translation;
    asset_meta(iasset).relative_rotation     = axang_rel;
    asset_meta(iasset).relative_rotmat       = R_rel;

    % accumulate
    all_assets = [all_assets; asset_final];
    s_assets   = [s_assets; a.s(iasset).*ones(size(asset_final,1),1)];
end

% ============================================================
% OUTPUT STRUCT
% ============================================================
TARG = [body; all_assets];
S.s  = [b.s.*ones(size(body,1),1); s_assets];
S.assets = asset_meta;

% ============================================================
% Visualization
% ============================================================
if doQC
    figure
    pcshow(body,b.s*ones(size(body,1),1)); hold on;
    pcshow(all_assets,s_assets);
    title(sprintf('Body with %d attached assets', nassets));
    colormap jet
    colorbar
    axis equal; view(3); drawnow;
    hold off
end

end
