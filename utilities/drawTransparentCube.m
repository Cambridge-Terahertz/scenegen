function drawTransparentCube(center, dims, color, alpha)
    % DRAWTRANSPARENTPATCH Draws a transparent 3D cube/cuboid
    % center: [x, y, z] center location
    % dims: [width, height, depth] (x, y, z dimensions)
    % color: [r, g, b] color (e.g., [0 0 1] for blue)
    % alpha: transparency (0 = invisible, 1 = opaque)

    % 1. Calculate corner vertices based on center and dimensions
    x = center(1); y = center(2); z = center(3);
    w = dims(1); h = dims(2); d = dims(3);

    % Vertices matrix (8 corners of the cube)
    v = [x-w/2, y-h/2, z-d/2;  % 1: Bottom-back-left
         x+w/2, y-h/2, z-d/2;  % 2: Bottom-back-right
         x+w/2, y+h/2, z-d/2;  % 3: Bottom-front-right
         x-w/2, y+h/2, z-d/2;  % 4: Bottom-front-left
         x-w/2, y-h/2, z+d/2;  % 5: Top-back-left
         x+w/2, y-h/2, z+d/2;  % 6: Top-back-right
         x+w/2, y+h/2, z+d/2;  % 7: Top-front-right
         x-w/2, y+h/2, z+d/2]; % 8: Top-front-left

    % 2. Define faces (connecting vertex indices)
    f = [1 2 6 5;  % Bottom face
         2 3 7 6;  % Right face
         3 4 8 7;  % Top face
         4 1 5 8;  % Left face
         1 2 3 4;  % Back face
         5 6 7 8]; % Front face

    % 3. Create Patch
    patch('Vertices', v, 'Faces', f, ...
          'FaceColor', color, ...
          'FaceAlpha', alpha, ...
          'EdgeColor', 'k', ... % Black edges
          'LineWidth', 1);

    % % 4. Formatting
    % axis equal;
    % grid on;
    % view(3);
    % xlabel('X'); ylabel('Y'); zlabel('Z');
    % 
    % % Ensure limits show the cube
    % xlim([x-w, x+w]); ylim([y-h, y+h]); zlim([z-d, z+d]);
end


