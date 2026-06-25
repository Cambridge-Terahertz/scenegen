function pts = read_xyz_binary(filename)
    fid = fopen(filename, 'rb');
    if fid < 0
        error('Could not open file: %s', filename);
    end

    data = fread(fid, inf, 'float32=>single');
    fclose(fid);

    if mod(numel(data), 3) ~= 0
        error('File size is not divisible by 12 bytes (x,y,z float32).');
    end

    pts = reshape(data, 3, []).';
end
