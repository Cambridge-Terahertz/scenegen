function xyz = import_xyz(xyz_file)

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

xyz = reshape(data, 3, []).';

