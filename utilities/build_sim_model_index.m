function model_index = build_sim_model_index(root_dir)

fprintf("=== Building Simulation Model Index ===\n");

d = dir(root_dir);
d = d([d.isdir]);
d = d(~ismember({d.name},{'.','..'}));

model_index = [];

for i = 1:numel(d)
    sim_folder = fullfile(root_dir, d(i).name);

    %% ---------------------------------------
    %  Load meta.json (height/BMI/etc)
    %% ---------------------------------------
    meta_file = fullfile(sim_folder, "meta.json");

    height_m = "";
    bmi = "";
    weight_kg = "";
    pose_type = "";
    gender = "";

    if isfile(meta_file)
        try
            raw = fileread(meta_file);
            meta = jsondecode(raw);

            height_m  = get_safe(meta, "height_m");
            bmi       = get_safe(meta, "bmi");
            weight_kg = get_safe(meta, "weight_kg");
            pose_type = get_safe(meta, "pose_type");
            gender    = get_safe(meta, "gender");

        catch ME
            fprintf("[ERROR] Failed to read %s: %s\n", meta_file, ME.message);
        end
    else
        fprintf("[WARN] No meta.json found in %s\n", sim_folder);
    end


    %% ---------------------------------------
    %  Load upsample_meta.json (models + timing)
    %% ---------------------------------------
    upsample_file = fullfile(sim_folder, "upsample_meta.json");

    model_list = "UNKNOWN";
    has_assets = false;
    contains_weapon = false;

    total_processing_time = "";
    mean_processing_time = "";
    max_processing_time = "";

    if isfile(upsample_file)
        try
            raw = fileread(upsample_file);
            up = jsondecode(raw);

            % Extract model names
            if isfield(up, "models") && ~isempty(up.models)
                names = strings(1, numel(up.models));
                proc_times = zeros(1, numel(up.models));

                for k = 1:numel(up.models)
                    % Model name
                    if isfield(up.models(k), "model")
                        names(k) = string(up.models(k).model);
                    else
                        names(k) = "UNKNOWN";
                    end

                    % Processing time
                    if isfield(up.models(k), "processing_time_sec")
                        proc_times(k) = up.models(k).processing_time_sec;
                    end
                end

                model_list = strjoin(names, ",");

                % Asset detection
                has_assets = any(names ~= "body.obj");

                % Weapon detection
                contains_weapon = any(contains(lower(names), ["gun","weapon","knife"]));

                % Timing summaries
                total_processing_time = sum(proc_times);
                mean_processing_time  = mean(proc_times);
                max_processing_time   = max(proc_times);
            end

        catch ME
            fprintf("[ERROR] Failed to read %s: %s\n", upsample_file, ME.message);
        end
    else
        fprintf("[WARN] No upsample_meta.json found in %s\n", sim_folder);
    end

    fprintf("[INFO] %s → models: %s\n", d(i).name, model_list);

    %% Store entry
    entry.folder    = d(i).name;
    entry.models    = model_list;
    entry.height_m  = height_m;
    entry.bmi       = bmi;
    entry.weight_kg = weight_kg;
    entry.pose_type = pose_type;
    entry.gender    = gender;

    entry.has_assets = has_assets;
    entry.contains_weapon = contains_weapon;

    entry.total_processing_time_sec = total_processing_time;
    entry.mean_processing_time_sec  = mean_processing_time;
    entry.max_processing_time_sec   = max_processing_time;

    model_index = [model_index; entry];
end


%% -------------------------
%  Write CSV summary
%% -------------------------
csv_file = fullfile(root_dir, "sim_model_index.csv");
fid = fopen(csv_file, "w");

fprintf(fid, ...
"folder,models,height_m,bmi,weight_kg,pose_type,gender,has_assets,contains_weapon,total_processing_time_sec,mean_processing_time_sec,max_processing_time_sec\n");

for i = 1:numel(model_index)
    fprintf(fid, "%s,%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%s\n", ...
        model_index(i).folder, ...
        model_index(i).models, ...
        string(model_index(i).height_m), ...
        string(model_index(i).bmi), ...
        string(model_index(i).weight_kg), ...
        string(model_index(i).pose_type), ...
        string(model_index(i).gender), ...
        model_index(i).has_assets, ...
        model_index(i).contains_weapon, ...
        string(model_index(i).total_processing_time_sec), ...
        string(model_index(i).mean_processing_time_sec), ...
        string(model_index(i).max_processing_time_sec));
end

fclose(fid);


%% -------------------------
%  Write JSON summary
%% -------------------------
json_file = fullfile(root_dir, "sim_model_index.json");
json_text = jsonencode(model_index, PrettyPrint=true);

fid = fopen(json_file, "w");
fwrite(fid, json_text, "char");
fclose(fid);

fprintf("[DONE] Index written to:\n  %s\n  %s\n", csv_file, json_file);

end


%% Helper: safe field extraction
function v = get_safe(s, field)
    if isfield(s, field)
        v = s.(field);
    else
        v = "";
    end
end
