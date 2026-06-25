model_src     = 'D:\model_exports\model_src\f25.1_001\xyz_0.5mm';
asset_src     = 'D:\model_exports\assets_src';
output_folder = 'D:\model_exports\26JUN19';

% Ensure output folder exists
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of frame*.xyz files
xyz_files = dir(fullfile(model_src, 'frame*.xyz'));

pose_idx = 1;

for k = 1:length(xyz_files)

    % ------------------------------------------------------------
    % Create pose folder
    % ------------------------------------------------------------
    pose_folder = fullfile(output_folder, sprintf('pose_%d', pose_idx));
    if ~exist(pose_folder, 'dir')
        mkdir(pose_folder);
    end

    % ------------------------------------------------------------
    % Create body/ and assets/ subfolders
    % ------------------------------------------------------------
    body_folder   = fullfile(pose_folder, 'body');
    assets_folder = fullfile(pose_folder, 'assets');

    if ~exist(body_folder, 'dir');   mkdir(body_folder);   end
    if ~exist(assets_folder, 'dir'); mkdir(assets_folder); end

    % ------------------------------------------------------------
    % Copy BODY model + JSON
    % ------------------------------------------------------------
    src_xyz = fullfile(model_src, xyz_files(k).name);
    dst_xyz = fullfile(body_folder, 'body.xyz');
    copyfile(src_xyz, dst_xyz);

    % Copy body.json (must exist in model_src)
    src_body_json = fullfile(model_src, 'body.json');
    if exist(src_body_json, 'file')
        copyfile(src_body_json, fullfile(body_folder, 'body.json'));
    else
        warning('Missing body.json in %s', model_src);
    end

    % ------------------------------------------------------------
    % Copy ASSET .xyz files
    % ------------------------------------------------------------
    asset_xyz_files = dir(fullfile(asset_src, '*.xyz'));

    for a = 1:length(asset_xyz_files)
        src_asset_xyz = fullfile(asset_src, asset_xyz_files(a).name);
        dst_asset_xyz = fullfile(assets_folder, asset_xyz_files(a).name);
        copyfile(src_asset_xyz, dst_asset_xyz);
    end

    % ------------------------------------------------------------
    % Copy the SINGLE shared assets.json
    % ------------------------------------------------------------
    shared_asset_json = fullfile(asset_src, 'assets.json');
    if exist(shared_asset_json, 'file')
        copyfile(shared_asset_json, fullfile(assets_folder, 'assets.json'));
    else
        warning('Missing assets.json in %s', asset_src);
    end

    fprintf('Created %s with body/ + assets/ (shared assets.json)\n', pose_folder);

    pose_idx = pose_idx + 1;
end
