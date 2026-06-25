% [files, products] = matlab.codetools.requiredFilesAndProducts('simulator.m');

addpath(genpath('utilities'))
addpath(genpath('D:\git_sandbox\model_gen\cpp\ForwardModelCudaProject\x64\Release'))
close all
clearvars
clc

filepath = 'D:\git_sandbox\model_gen';

param_file  = fullfile(filepath,'sim_params_mixamo.json');
proc_file   = fullfile(filepath,'process_list_mixamo.json');

% ============================================================
% LOAD PROCESS LIST
% ============================================================
fprintf('[%s] processing this file: %s...\n',datetime("now"),proc_file)
fid = fopen(proc_file,'r');
str = textscan(fid,'%c');
fclose(fid);
j = jsondecode(str{1}');

proc_list = j.proc_list;
nfiles = numel(proc_list);
useCUDA = true;
useAssets = false;

% ============================================================
% CREATE MASTER RUN FOLDER (AUTO-INCREMENT)
% ============================================================
date_tag = datestr(now,'ddmmmyy');   % e.g. '26JUN05'
date_tag = upper(date_tag);

base_dir = 'D:\outputs\sim_results';                 % parent folder
pattern = sprintf('%s_run_', date_tag);

% Find existing run folders
d = dir(fullfile(base_dir, [pattern '*']));
if isempty(d)
    run_idx = 1;
else
    % Extract numeric suffix
    nums = [];
    for k = 1:numel(d)
        tok = regexp(d(k).name, [pattern '(\d+)'], 'tokens');
        if ~isempty(tok)
            nums(end+1) = str2double(tok{1}{1});
        end
    end
    if isempty(nums)
        run_idx = 1;
    else
        run_idx = max(nums) + 1;
    end
end

run_folder = fullfile(base_dir, sprintf('%s_run_%03d', date_tag, run_idx));
fprintf('[%s] Creating master run folder: %s\n', datetime("now"), run_folder);
mkdir(run_folder);

% ============================================================
% RUN SIMULATIONS (WITH AND WITHOUT ASSETS)
% ============================================================
for ifile = 1:1;nfiles
    this_proc = proc_list(ifile);

    fp       = this_proc.filepath;
    pose_nr  = sprintf('pose_%d', this_proc.pose);
    model_path = fullfile(fp, pose_nr);

    fprintf('\n============================================================\n');
    fprintf('[%s] Running simulations on: %s\n', datetime("now"), model_path);
    fprintf('[%s] Processing model %d of %d\n', datetime("now"), ifile, nfiles);
    fprintf('============================================================\n');

    % ------------------------------------------------------------
    % Run twice: without assets, then with assets
    % ------------------------------------------------------------
    for useAssets = [0 1]

        if useAssets
            mode_label = 'WITH_ASSETS';
        else
            mode_label = 'NO_ASSETS';
        end

        fprintf('\n[%s] Starting simulation (%s)\n', datetime("now"), mode_label);

        % Run simulation
        sim_out_folder = simulator_mixamo_flow(param_file, model_path, useCUDA, useAssets);
        close all
        % Move simulation output folder into master run folder
        [~, sim_name] = fileparts(sim_out_folder);

        % Append mode label to folder name
        sim_name = sprintf('%s_%s', sim_name, mode_label);

        new_loc = fullfile(run_folder, sim_name);

        fprintf('[%s] Moving simulation output to: %s\n', datetime("now"), new_loc);
        fprintf('[%s] Folder link: <a href="matlab: winopen(''%s'')">%s</a>\n', ...
            datetime("now"), new_loc, new_loc);

        movefile(sim_out_folder, new_loc);

        o{ifile, useAssets+1} = new_loc;
    end
end

fprintf('Done\n');
