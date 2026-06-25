% place_asset_on_waist_simplified_batch
% For each body frame, attach ALL assets and save ONE combined file.
addpath(genpath('D:\git_sandbox\model_gen\utilities'))
close all
clearvars
clc

datapath    = 'D:\model_exports\f25.1_001\xyz_0.5mm';
body_files  = dir(fullfile(datapath,'frame_*.xyz'));
asset_files = {'phone.xyz','gun.xyz'};

nbod    = numel(body_files);
% nbod = 10;
doSaveFiles = false;
nassets = numel(asset_files);
at = [0 0 90; 90 0 90]*pi/180;
ap = [-.2 -.5 -.2;.05 -0.5 -0.1];
ac ='yr';
% Per-asset tilt and extra translation
tilt_angle_deg    = [0 3];                 % tilt per asset
extra_translation = [0 0 0; 0 0 0];       % per asset [dx dy dz]

for ibody = 1:nbod

    % -----------------------------
    % Load and transform body ONCE per frame
    % -----------------------------
    body_file = fullfile(datapath, body_files(ibody).name);
    body      = read_xyz_binary(body_file);

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
    
    S = [];
    all_assets = [];   % will hold ALL attached assets for this body

    for iasset = 1:nassets

        % -----------------------------
        % Load asset
        % -----------------------------
        asset_file = fullfile(datapath, asset_files{iasset});
        asset      = read_xyz_binary(asset_file);
        xt = at(iasset,1);
        yt = at(iasset,2);
        zt = at(iasset,3);
        % Pre-rotate asset
        asset = asset - mean(asset);
        TM_a  = transform(xt,yt,zt);
        asset = [asset ones(size(asset,1),1)];
        asset = TM_a*asset.';
        asset = asset(1:3,:).';

        % -----------------------------
        % Pre-place asset near body (randomized per asset)
        % -----------------------------
        p = [0.1*rand, 0, 0.1*rand];
        
        preplacement = ap(iasset,:) + p;
        asset = asset + preplacement;

        % -----------------------------
        % 1. Asset attachment point
        % -----------------------------
        asset_attach = mean(asset,1);

        % -----------------------------
        % 2. Closest body point
        % -----------------------------
        dists = vecnorm(body - asset_attach, 2, 2);
        [~, idx_min] = min(dists);
        body_hit = body(idx_min, :);

        % -----------------------------
        % 3. Local surface normal at body_hit
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
        % 4. Thickness-based offset
        % -----------------------------
        proj      = asset * normal';
        thickness = max(proj) - min(proj);
        offset    = 0.25 * thickness + 0.003;

        target_point    = body_hit + offset * normal;
        translation_vec = target_point - asset_attach;

        asset_moved = asset + translation_vec;

        % -----------------------------
        % 5. FINAL LOCAL-AXIS TILT (per asset)
        % -----------------------------
        [coeff_asset, ~] = pca(asset_moved - mean(asset_moved,1));
        local_axis = coeff_asset(:,1)';   % local X axis

        tilt_angle = deg2rad(tilt_angle_deg(iasset)+.3*randn(1));
        tilt_axis  = local_axis / norm(local_axis);
        R_tilt     = axang2rotm([tilt_axis tilt_angle]);

        asset_final = (R_tilt * (asset_moved - asset_attach)')' + asset_attach;
        asset_final = asset_final + extra_translation(iasset,:);

        % accumulate this asset for this body
        all_assets = [all_assets; asset_final];
        S = [S; (1+iasset)*ones(size(asset_final,1),1)];

    end

    % -----------------------------
    % 6. Save ONE combined file per body (with ALL assets)
    % -----------------------------
    out_name = sprintf('out_body%03d_allassets.xyz', ibody);
    out_path = fullfile(datapath, out_name);

    combined = [body; all_assets];
    if doSaveFiles
        write_xyz_binary(out_path, combined);
    end

    % -----------------------------
    % Visualization
    % -----------------------------
    figure(1); clf;
    pcshow(body,ones(size(body,1),1)); hold on;
    pcshow(all_assets,S);
    title(sprintf('Body %d with %d attached assets', ibody, nassets));
    axis equal; view(3); drawnow;
    if ~doSaveFiles
        drawnow
    end

end

% -----------------------------
% Helpers
% -----------------------------
function pts = read_xyz_binary(filename)
fid = fopen(filename, 'rb');
data = fread(fid, inf, 'float32=>single');
fclose(fid);
pts = reshape(data, 3, []).';
end

function write_xyz_binary(filename, pts)
fid = fopen(filename, 'wb');
fwrite(fid, single(pts.'), 'float32');
fclose(fid);
end
