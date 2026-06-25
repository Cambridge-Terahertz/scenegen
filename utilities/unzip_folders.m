% 1. Define the root directory where your zipped folders are located
root_dir = 'D:/Data for Scott/2026FEB_joe_scans'; % e.g., 'C:\Users\YourUser\Documents\Project'

% 2. Define the destination directory where you want files to be extracted
% If you want each zip to extract into a folder with the same name as the zip file
% within a single main output directory, you can set the output base path.
output_base_dir = 'D:/Data for Scott/2026FEB_joe_scans/output'; % e.g., 'C:\Users\YourUser\Documents\Extracted'

% Create the base output directory if it doesn't exist
if ~exist(output_base_dir, 'dir')
    mkdir(output_base_dir);
end

% 3. Find all .zip files recursively within the root directory
% The '**' wildcard searches all subfolders
% dinfo = dir(fullfile(root_dir, '**', '*.zip'));
dinfo = dir(fullfile(root_dir, '*.zip'));

% 4. Iterate through each found zip file and unzip it
for i = 1:numel(dinfo)
    % Get the full path to the current zip file
    zip_file_path = fullfile(dinfo(i).folder, dinfo(i).name);

    % Get the name of the zip file without the extension, to use for the output folder name
    [~, zip_name, ~] = fileparts(dinfo(i).name);

    % Define the specific output folder for the current zip file
    current_output_folder = fullfile(output_base_dir, zip_name);

    % Create the specific output folder if it doesn't exist
    if ~exist(current_output_folder, 'dir')
        mkdir(current_output_folder);
    end
    
    % Unzip the file into the specified output folder
    disp(['Unzipping: ' zip_file_path ' to ' current_output_folder]);
    unzip(zip_file_path, current_output_folder);
    disp('Unzipping complete.');
end

disp('All zip files processed.');
