function [res, V] = load_3d_model(p)
t1 = tic;
fn = p.model; % full path to ply files
if contains(fn,'cylinder')
    p.model_orientation = [1 2 3];
end

if ~isfield(p,"model_orientation")
    p.model_orientation = [3 1 2];
end

tx = p.tx;
% rx = p.rx;
% rots = p.rotations;
% f = p.freq;
HPR_Fac = p.HPR_fac;
model_scale_fac = p.model_scale_fac; %3.5
% nrots = length(rots);

% =========================================================================
% tx = [2-.001i 0.0 .12]; % depth width height
% rx = [2-.001i 0.0 0];

displayMesh = false;
% f = 275e9;  % frequency used for calculating beampattern


%==========================================================================
% Receive Plane
%==========================================================================
% L = p.L;
% y1 = -L; y2 = L; % width
% z1 = -L; z2 = L; % height
% x1 = 2; x2 = 2; % depth



%==========================================================================
% load 3d model, lets try using faceted model
%==========================================================================

% filename = 'Male.ply';
% filename = 'Male_Large.ply';
% filename = 'Small_Male.ply'; % load small model for development, faster and easier to see what is happening
% filename = 'Very_Small_Male.ply';
% filename = 'Medium_Male.ply';

lidarToolboxAvailable = contains(lower(struct2array(ver)), 'lidar');

% fprintf('Lidar Toolbox: %d\n',lidarToolboxAvailable);

if lidarToolboxAvailable
	fprintf('[%s] Using Lidar Toolbox.\n',datetime("now"))
	for iobjs = 1:numel(p.model)
		fprintf('[%s] Loading model: %s\n',datetime("now"),fn{iobjs})
		mesh0{iobjs} = readSurfaceMesh(p.model{iobjs});
        
	end
	
else
	fprintf('\n[%s] **** LIDAR TOOLBOX NOT AVAILABLE **** ===> using work around for loading mesh.\n\n',datetime("now"))
	for iobjs = 1:numel(p.model)
		fprintf('[%s] Loading model: %s\n',datetime("now"),fn{iobjs})
		temp = plyread(p.model{iobjs}); % # <THIRD PARTY CODE>
        mesh0{iobjs}.Vertices = [temp.vertex.x temp.vertex.y temp.vertex.z];
        mesh0{iobjs}.Faces = cell2mat(temp.face.vertex_indices);
	end
end


% subdivide(mesh0{1},"midpoint-split",2);
% subdivide(mesh0{2},"midpoint-split",2);


F_ = mesh0{1}.Faces;
V0_ = mesh0{1}.Vertices;
S = ones(size(V0_,1),1).*p.s(1);
for iobj = 2:numel(p.model)
    V0_ = [V0_; mesh0{iobj}.Vertices];
    Ftemp = mesh0{iobj}.Faces + size(mesh0{iobj-1}.Vertices, 1); % Adjust face indices for the second mesh
    F_ = [F_; Ftemp];
    S = [S; ones(size(mesh0{iobj}.Vertices, 1),1).*p.s(iobj)];
end



N = size(S,1);

if displayMesh
    surfaceMeshShow(mesh0);
end
%==========================================================================
% Define transmitter and rx locations and a gaussian
% beamwidth using complex positions
% =========================================================================
% tx = [2-.001i 0.0 .12]; % depth width height
% rx = [2-.001i 0.0 0];

%==========================================================================
% Scale,center and change model matrix columns indices to orient
%  model up-down
%==========================================================================


F = F_;
% V0 =V0_;
V0 = V0_(:,p.model_orientation)./model_scale_fac;
mn = mean(V0);
mx = max(V0(:,3));
mns = min(V0(:,3));
center = (mx+mns)/2;
mn(3)=center;
V0 = V0-mn;
TM = transform(p.rotations*pi/180,0,0);

V =[V0 ones(size(V0,1),1)];
V = TM*V.';
V = V.';
V = V(:,1:3);


%==========================================================================
% create mesh object so we can utilize built in surface norm calculations
%==========================================================================

%==========================================================================
% Calculate the distance from transmitter to center of a patch (barycenter)
%==========================================================================

doBC = false;
if doBC
    fprintf('[%s] Calculating barycenters...\n',datetime("now"))
    nfaces = size(F,1);
    bc = zeros(nfaces,3);
    s = bc;
    for ii = 1:nfaces
        bc(ii,:) = mean(V(F(ii,:),:));
        s(ii) =  mean(S(F(ii,:),:));
        % if mod(ii,50000)==0
        %     fprintf('[%s] [tx %d of %d] Calculating barycenters %d of %d...\n',datetime("now"),itx,ntx,ii,nfaces)
        % end
    end
else
    fprintf('[%s] Not using barycenters, using vertices...\n',datetime("now"))
    bc = V;
    s = S;
	if p.resample.doResample
		resample_nr = p.resample.resample_nr;
		fprintf(sprintf('[%s] ***** TESTING ******* ====> resampling surface: %d\n',datetime("now"),resample_nr))
		pc = pointCloud(bc);

		mesh = pc2surfacemesh(pc,"ball-pivot");
		pc = mesh2pc(mesh,'SamplingMethod','UniformSampling','NumPoints',resample_nr);
		bc = pc.Location;
		s = ones(size(bc,1),1);
	end


end

%==========================================================================
% Create a point cloud object with barycenters so we can use point cloud
% object hidden point removal method to remove model facets which should
% not be seen from given view point. This also speeds up calculations by
% removing data which doesn't need to be processed.
%==========================================================================
fprintf('[%s] Removing hidden points...\n',datetime("now"))
bc = bc+[0 2 0];
%pc = pointCloud(bc);
%[pc, indc] = removeHiddenPoints(pc,real(tx),"RadiusScale",HPR_Fac);
[indc] = hpr(bc,real(tx),HPR_Fac);

res.bc = bc(indc,:);
res.s = s(indc);
res.model_original_size = N;
res.model_size_after_hpr = size(res.bc,1);
fprintf('[%s] Model size %d ===> %d\n',datetime("now"),size(res.bc,1),N);
fprintf('[%s] This process: Loading data and applying HPR: %0.1f [sec]\n',datetime("now"),toc(t1))





