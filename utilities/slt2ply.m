% convert stl to ply using meshlab server
src = 'D:\models\exported_poses_20set_1mm';
out = fullfile(src,'ply');
mkdir(out);


% Define the input STL file names
files = dir(fullfile(src,'*.stl'));


meshlab_path = '"C:\Program Files\VCG\MeshLab\meshlab.exe"'; 
nfiles = numel(files);

% Loop through each file
for ifile = 1:nfiles
    infiles = files(ifile).name;
    [path, name, ext] = fileparts(infiles);
    outfile = fullfile(out, sprintf('%s.ply',name));
    
    % Construct the command to run meshlabserver
    % -i for input, -o for output, -om to specify output mesh attributes
    % 'vc vn' would include vertex colors and normals, if applicable
    command = sprintf('%s -i "%s" -o "%s"', meshlab_path, infiles, outfile);
    
    % Execute the command using the system function
    [status, cmdout] = system(command);
    
    % Check for success
    if status == 0
        disp(['Converted ', infiles, ' to ', outfile]);
    else
        disp(['Error converting ', infiles]);
        disp(cmdout);
    end
end
