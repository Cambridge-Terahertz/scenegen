function TM = transform(alpha, beta, gamma, dx,dy,dz,s1,s2,s3)
% Cretae a transformation matrix: rotations, scaling and translation.

% alpha = rotation around x axis (in radians)
%  beta = roation around y axis (in radians)
% gamma = rotation around z axis (in radians)
%    dx = translation along x axis
%    dy = translation along y axis
%    dz = translation along z axis
%    s1 = scaling along x axis
%    s2 = scaling along y axis
%    s3 = scaling along z axis

if ~exist('s1','var')
    s1 = 1;
end
if ~exist('s2','var')
    s2 = 1;
end
if ~exist('s3','var')
    s3 = 1;
end
if ~exist('dx','var')
    dx = 0;
end
if ~exist('dy','var')
    dy = 0;
end
if ~exist('dz','var')
    dz = 0;
end

%==========================================================================
% Rotation around z-axis
%==========================================================================
RZ =[cos(alpha) -sin(alpha)     0       0; ...
     sin(alpha) cos(alpha)      0       0; ...
     0            0             1       0;...
     0            0             0       1];
%==========================================================================
% Rotation around y-axis
%==========================================================================
RY = [cos(beta)   0  sin(beta)  0; ...
      0           1  0          0; ...
      -sin(beta)  0  cos(beta)  0;...
      0           0  0          1];
%==========================================================================
% Rotation around x-axis
%========================================================================== 
RX = [1  0           0            0 ;...
      0  cos(gamma)  -sin(gamma)  0 ; ...
      0  sin(gamma)  cos(gamma)   0; ...
      0  0           0            1];
 
%==========================================================================
% Scaling and Translations
%========================================================================== 
T = [s1      0     0    dx; ...
     0      s2     0    dy; ...
     0       0    s3    dz
     0       0     0    1];
 
%==========================================================================
% Total Transformation Matrix
%==========================================================================
TM = RX*RY*RZ*T;