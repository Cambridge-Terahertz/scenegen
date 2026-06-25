function [vertices, faces] = drawCube(width, height, depth, center,color, FaceAlpha)
% DRAWCUBE Creates the vertices and faces of a 3D cube.
%   [vertices, faces] = CREATECUBE(width, height, depth, center)
%   defines a cube with the specified width, height, and depth,
%   centered at the 'center' coordinates [cx, cy, cz].
%
%   Inputs:
%     width  - Scalar, the width of the cube along the x-axis.
%     height - Scalar, the height of the cube along the y-axis.
%     depth  - Scalar, the depth of the cube along the z-axis.
%     center - 1x3 vector [cx, cy, cz], the coordinates of the cube's center.
%
%   Outputs:
%     vertices - 8x3 matrix, coordinates of the 8 vertices of the cube.
%     faces    - Nx4 matrix, indices of vertices forming the cube's faces.

% Calculate half dimensions
hw = width / 2;
hh = height / 2;
hd = depth / 2;

% Calculate vertex coordinates relative to the origin
% The order of vertices is important for defining faces correctly
vertices_relative = [
    -hw, -hh, -hd;  % 1
     hw, -hh, -hd;  % 2
     hw,  hh, -hd;  % 3
    -hw,  hh, -hd;  % 4
    -hw, -hh,  hd;  % 5
     hw, -hh,  hd;  % 6
     hw,  hh,  hd;  % 7
    -hw,  hh,  hd   % 8
];

% Translate vertices to the specified center
vertices = vertices_relative + repmat(center, 8, 1);

% Define the faces of the cube using vertex indices
faces = [
    1, 2, 6, 5;  % Bottom face
    2, 3, 7, 6;  % Right face
    3, 4, 8, 7;  % Top face
    4, 1, 5, 8;  % Left face
    1, 2, 3, 4;  % Front face (or back depending on perspective)
    5, 6, 7, 8   % Back face (or front depending on perspective)
];
patch('Vertices', vertices, 'Faces', faces, 'FaceColor', color, 'FaceAlpha', FaceAlpha);
% % Optional: Plot the cube if no output arguments are requested
% if nargout == 0
%     patch('Vertices', vertices, 'Faces', faces, 'FaceColor', 'blue', 'FaceAlpha', 0.5);
%     xlabel('X');
%     ylabel('Y');
%     zlabel('Z');
%     axis equal;
%     view(3);
%     grid on;
%     title('3D Cube');
% end

end