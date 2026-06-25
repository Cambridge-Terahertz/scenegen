filename = 'mesh.obj'; % Replace with your actual file name

% Open the file for reading
fileID = fopen(filename, 'r');

% Initialize arrays to hold vertices, texture coordinates, and normals
vertices = [];
textureCoords = [];
normals = [];

% Read the file line by line
while ~feof(fileID)
    line = fgetl(fileID);
    if startsWith(line, 'v ') % Vertex data
        vertex = sscanf(line(3:end), '%f')';
        vertices = [vertices; vertex];
    elseif startsWith(line, 'vt ') % Texture coordinate data
        texCoord = sscanf(line(4:end), '%f')';
        textureCoords = [textureCoords; texCoord];
    elseif startsWith(line, 'vn ') % Normal data
        normal = sscanf(line(4:end), '%f')';
        normals = [normals; normal];
    end
end

% Close the file
fclose(fileID);

% Display the imported data
disp('Vertices:');
disp(vertices);
disp('Texture Coordinates:');
disp(textureCoords);
disp('Normals:');
disp(normals);