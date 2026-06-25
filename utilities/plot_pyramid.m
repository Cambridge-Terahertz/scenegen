function p = plot_pyramid(varargin)

% Default Settings
baseSize = 2; % Side length of the square base
height = 3;   % Height of the pyramid
% --- Define Pyramid Origin and Orientation ---
origin = [1, 2, 0]; % Center of the pyramid's base
orientation = [0, 1, 1]; % Desired direction for the pyramid's apex (example: pointing along the y=z line)
doQC = true;
color = 'b';


for iarg = 1:2:nargin
    key = varargin{iarg};
    val = varargin{iarg+1};

    switch key
        case 'baseSize'
            baseSize = val; % Update baseSize with the provided value
        case 'height'
            height = val; % Update height with the provided value
        case 'origin'
            origin = val;
        case 'orientation'
            orientation = val; % Update orientation with the provided value
        case 'doQC'
            doQC = val; % Update doQC with the provided value
        case 'fh'
            fh = val;
        case 'color'
            color = val;
        case 'ax'
            ax = val;
    end
end


% --- Define Base Vertices (centered at origin, initially aligned with xy plane) ---
vertices_base = [
    -baseSize/2, -baseSize/2, 0;  % Vertex 1 (bottom left)
    baseSize/2, -baseSize/2, 0;   % Vertex 2 (bottom right)
    baseSize/2, baseSize/2, 0;    % Vertex 3 (top right)
    -baseSize/2, baseSize/2, 0    % Vertex 4 (top left)
];

% --- Define Apex Vertex (initially aligned with positive z-axis) ---
apex = [0, 0, height];

% --- Combine all vertices ---
vertices = [vertices_base; apex]; %

% --- Define Faces for the Pyramid ---
% Each row in `faces` represents a face and contains the indices of the vertices that form that face.
% The order of indices matters for face orientation (clockwise or counterclockwise for normal vectors)
faces = [
    1, 2, 3, 4; % Base (bottom face)
    1, 2, 5, NaN; % Face 1 (side) - connect vertices 1, 2, apex
    2, 3, 5, NaN; % Face 2 (side) - connect vertices 2, 3, apex
    3, 4, 5, NaN; % Face 3 (side) - connect vertices 3, 4, apex
    4, 1, 5, NaN  % Face 4 (side) - connect vertices 4, 1, apex
]; %




% --- Normalize the orientation vector ---
orientation = orientation / norm(orientation);

% --- Calculate Rotation Matrix to Align with Orientation Vector ---
% Initial pyramid's apex is at [0, 0, height], so we rotate from [0, 0, 1] to the orientation vector.
initial_direction = [0, 0, 1]; % The direction of the pyramid's default orientation
rotation_axis = cross(initial_direction, orientation);
rotation_angle = acos(dot(initial_direction, orientation));

% Use makehgtform to create a rotation matrix
if norm(rotation_axis) > 1e-6 % Avoid division by zero if vectors are parallel
    R = makehgtform('axisrotate', rotation_axis, rotation_angle); %
else % If vectors are parallel (e.g., initial_direction and orientation_vector are same or opposite)
    if dot(initial_direction, orientation) < 0 % Opposite direction, rotate 180 degrees
        R = makehgtform('xrotate', pi); % Rotate 180 degrees around x-axis
    else % Same direction, no rotation needed
        R = eye(4); % Identity matrix
    end
end

% --- Apply Rotation and Translation ---
% Translate to the desired origin
T = makehgtform('translate', origin); %

% Combine transformations (translation then rotation is generally the order for applying to points)
% Note: makehgtform applies transforms right-to-left, so translation first followed by rotation
transform_matrix = R * T; %

% Create a figure and axes
if doQC
figure(fh);

% Create a hgtransform object for the pyramid
h = hgtransform('Parent', ax); %

% Plot the pyramid using patch
pyramid_handle = patch('Parent', h, 'Faces', faces, 'Vertices', vertices, 'FaceColor', color, 'FaceAlpha', 0.7); %

% Apply the transformation to the hgtransform object
set(h, 'Matrix', transform_matrix); %

% Optionally, plot the orientation vector to visualize it
% hold on;
% quiver3(origin(1), origin(2), origin(3), ...
%         orientation(1), orientation(2), orientation(3), ...
%         'r', 'LineWidth', 2, 'DisplayName', 'Orientation Vector');
% hold off;
end
