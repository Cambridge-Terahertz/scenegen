function output_folder = simulator_mixamo_flow(param_file, model_path, useCUDA, useAssets)

if nargin < 4
    useAssets = true;
end

if nargin<3
    useCUDA = false;
end
T0 = tic;
addpath(genpath('./utilities'))

fprintf('\n\n%s\n', repmat('*',[1 100]))
fprintf('[%s]                    **** STARTING SIMULATION ****\n', datetime("now"))
fprintf('%s\n\n', repmat('*',[1 100]))

% Load parameters
params = load_params(param_file);

% Override model path
params.object.filepath = model_path;

% Create output folder
tag = sprintf('%s-ra-sim', datetime("now","format","yyMdhms"));
output_folder = fullfile(params.output_loc, tag);


%+===============================================================================
%% CREATE OUTPUT DIRECTORY
%+===============================================================================
if ~exist(output_folder,'dir')
    fprintf('[%s] Creating output folder: %s\n', datetime("now"),output_folder)
    mkdir(output_folder)
end

%================================================================================
%% START COMMAND LINE LOGGING (diary)
%================================================================================
try
    log_file = fullfile(output_folder, 'command_line_log.txt');
    fprintf('[%s] Saving command line output to: %s\n', datetime("now"), log_file);

    if get(0,'Diary')
        diary off;   % close any existing diary session
    end

    diary(log_file);
    diary on;

catch ME
    fprintf('[%s] ERROR starting diary logging: %s\n', datetime("now"), ME.message);
end


%================================================================================
%% COPY sim_params.json INTO OUTPUT FOLDER
%================================================================================
try
    src_param_file = param_file;  % original sim_params.json
    dst_param_file = fullfile(output_folder, filename);

    fprintf('[%s] Copying parameter file to output folder: %s\n', ...
        datetime("now"), dst_param_file);

    copyfile(src_param_file, dst_param_file);
catch ME
    fprintf('[%s] ERROR copying sim_params.json: %s\n', datetime("now"), ME.message);
end

%================================================================================
%% COPY run_simulator_script INTO OUTPUT FOLDER
%================================================================================
try
    % Determine the script name and location
    script_name = 'run_simulator_script.m';   % or .txt if that's your version
    script_name_out = '_run_simulator_script.m';
    script_src  = fullfile(filepath, script_name);

    % Destination inside the output folder
    script_dst  = fullfile(output_folder, script_name_out);

    fprintf('[%s] Copying run_simulator_script to output folder: %s\n', ...
        datetime("now"), script_dst);

    copyfile(script_src, script_dst);

catch ME
    fprintf('[%s] ERROR copying run_simulator_script: %s\n', ...
        datetime("now"), ME.message);
end


%================================================================================
%% COPY upsample_meta.json INTO OUTPUT FOLDER
%================================================================================
try
    upsample_meta_src = fullfile(params.object.filepath, 'upsample_meta.json');
    upsample_meta_dst = fullfile(output_folder, 'upsample_meta.json');

    fprintf('[%s] Copying upsample_meta.json to output folder: %s\n', ...
        datetime("now"), upsample_meta_dst);

    copyfile(upsample_meta_src, upsample_meta_dst);

catch ME
    fprintf('[%s] ERROR copying upsample_meta.json: %s\n', ...
        datetime("now"), ME.message);
end


%================================================================================
%% COPY meta.json (parent directory) INTO OUTPUT FOLDER
%================================================================================
try
    parent_dir = fileparts(params.object.filepath);
    meta_src = fullfile(parent_dir, 'meta.json');
    meta_dst = fullfile(output_folder, 'meta.json');

    fprintf('[%s] Copying meta.json to output folder: %s\n', ...
        datetime("now"), meta_dst);

    copyfile(meta_src, meta_dst);

catch ME
    fprintf('[%s] ERROR copying meta.json: %s\n', ...
        datetime("now"), ME.message);
end




% ======================================================================
proc_type = check_proc_type(params);
logging_frequency = params.options.logging_frequency;

c = 299792458;
nf = params.freq.nf;
f1 = params.freq.f1;
f2 = params.freq.f2;

f = linspace(f1,f2,nf);
lambda = c/f(end);


% if doCUDA
%     % [xx,yy,zz,xs0,ys0,RX0,TXS,ny,nx,dx,dy,nr_elements] = ra_geom_cuda(params);
%     [~,~,~,xs0,ys0,RX0,TXS,ny,nx,dx,dy,nr_elements] = ra_geom(params);
% else

    [~,~,~,xs0,ys0,RX0,TXS,ny,nx,dx,dy,nr_elements] = ra_geom(params);
% end

% ======================================================================
% Load XYZ scatterers + reflectivity
% ======================================================================
t_load_model = tic;
fprintf('[%s] Loading scatterer model...\n', datetime("now"));

datapath = params.object.filepath;

% -----------------------------
% BODY ALWAYS LOADED
% -----------------------------
b.files  = fullfile(datapath,'body','body.xyz');
fid = fopen(fullfile(datapath,'body','body.json'));
str = textscan(fid,'%c'); fclose(fid);
str = str{1}.';
b_params = jsondecode(str);
b.s = b_params.body.s;

% ------------------------------------------------------------
% Compute BODY bounding box (before asset placement)
% ------------------------------------------------------------
body_pts = read_xyz_binary(b.files);   % your existing xyz reader
body_bbox_min = min(body_pts,[],1);
body_bbox_max = max(body_pts,[],1);
body_center   = mean(body_pts,1);

b.bbox_min = body_bbox_min;
b.bbox_max = body_bbox_max;
b.center   = body_center;


fprintf('[%s] Loaded BODY: %s (s = %.3f)\n', ...
    datetime("now"), b.files, b.s);

% -----------------------------
% ASSETS OPTIONAL
% -----------------------------
if useAssets
    fprintf('[%s] Assets ENABLED\n', datetime("now"));

    temp = dir(fullfile(datapath,'assets','*.xyz'));
    a.files = fullfile(datapath,'assets',{temp.name});

    % Load asset metadata JSON
    fid = fopen(fullfile(datapath,'assets','assets.json'),'r');
    raw = fread(fid, inf, 'uint8=>char')'; fclose(fid);
    j = jsondecode(raw);

    n = numel(temp);
    s_vals = zeros(n,1);

    fprintf('[%s] Found %d asset files:\n', datetime("now"), n);

    for k = 1:n
        [~, base, ~] = fileparts(temp(k).name);

        if isfield(j, base)
            s_vals(k) = j.(base).s;
            fprintf('   • %s  (s = %.3f)\n', temp(k).name, s_vals(k));
        else
            s_vals(k) = NaN;
            fprintf('   • %s  (NO MATCHING ENTRY IN assets.json)\n', temp(k).name);
        end
    end

    a.s = s_vals;

    % Asset transforms
    a.prerotations = [90 0 90; 0 0 90]*pi/180;
    a.preplacement = [-.2 -.5 -.2; .05 -0.5 -0.1];
    a.final_tilt   = [0 3];
    a.final_trans  = [0 0 0; 0 0 0];

    doQC = true;
    [TARG, SS] = place_assets(b, a, doQC);
 % ============================================================
% SAVE ASSET LOCATIONS + BOUNDING BOXES
% ============================================================
% ============================================================
% SAVE BODY + ASSET METADATA TO JSON
% ============================================================
% ============================================================
% SAVE BODY + ASSET METADATA TO PRETTY JSON
% ============================================================
% ============================================================
% SAVE BODY + ASSET METADATA TO PRETTY JSON
% ============================================================
% ============================================================
% SAVE BODY + ASSET METADATA TO JSON (PrettyPrint)
% ============================================================
if useAssets
    try
        assets_info = struct();
        assets_info.timestamp   = char(datetime("now"));
        assets_info.model_path  = datapath;
        assets_info.sim_tag     = tag;

        % -------------------------
        % BODY INFORMATION
        % -------------------------
        assets_info.body = struct();
        assets_info.body.file        = SS.body.file;
        assets_info.body.scale       = SS.body.scale;
        assets_info.body.bbox_min    = SS.body.bbox_min;
        assets_info.body.bbox_max    = SS.body.bbox_max;
        assets_info.body.center      = SS.body.center;

        % If final body bbox was computed later in the pipeline:
        if isfield(SS.body, "bbox_min_final")
            assets_info.body.bbox_min_final = SS.body.bbox_min_final;
            assets_info.body.bbox_max_final = SS.body.bbox_max_final;
            assets_info.body.center_final   = SS.body.center_final;
        end

        % -------------------------
        % ASSET INFORMATION
        % -------------------------
        nA = numel(SS.assets);
        asset_list = cell(1,nA);

        for k = 1:nA
            A = SS.assets(k);

            entry = struct();
            entry.name                   = A.name;
            entry.file                   = A.file;
            entry.semantic_label         = A.semantic_label;

            entry.translation            = A.translation;
            entry.rotation_axis_angle    = A.rotation;

            entry.bbox_min               = A.bbox_min;
            entry.bbox_max               = A.bbox_max;

            entry.attach_body            = A.attach_body;
            entry.normal                 = A.normal;

            entry.relative_translation   = A.relative_translation;
            entry.relative_rotation      = A.relative_rotation;
            entry.relative_rotmat        = A.relative_rotmat;

            asset_list{k} = entry;
        end

        assets_info.assets = asset_list;

        % -------------------------
        % WRITE JSON FILE
        % -------------------------
        fn = fullfile(output_folder, 'assets_locs.json');
        jsonStr = jsonencode(assets_info, "PrettyPrint", true);

        fid = fopen(fn,'w');
        fwrite(fid, jsonStr, 'char');
        fclose(fid);

        fprintf('[%s] Saved pretty‑printed asset metadata JSON: %s\n', ...
            datetime("now"), fn);

    catch ME
        fprintf('[%s] ERROR writing assets_locs.json: %s\n', ...
            datetime("now"), ME.message);
    end
end




else
    fprintf('[%s] Assets DISABLED — using BODY ONLY\n', datetime("now"));

    % Load only the body.xyz
    [TARG, SS] = process_body_only(b.files, b.s);
    fprintf('[%s] Loaded %d body scatterers\n', datetime("now"), size(TARG,1));
end


% TARG = TARG(1:100:end,:);
% S = S(1:100:end);

t_load_model = toc(t_load_model);

% ======================================================================
% Transform geometry
% ======================================================================
% TARG = TARG(:,[3 1 2]);
% TM = transform(-90*pi/180,0,0);

mn = mean(TARG);
mx = max(TARG(:,3));
mns = min(TARG(:,3));
center = (mx+mns)/2;
mn(3) = center;

TARG = TARG - mn;
% TARG = [TARG ones(size(TARG,1),1)];
% TARG = TM*TARG.';
% TARG = TARG.';
% TARG = TARG(:,1:3);
TARG = TARG + [0 2 0];

inds = hpr(double(TARG),[0 -1 0],2.5);
TARG = TARG(inds,:);
S = SS.s(inds).';

% ======================================================================
% Remove points closer than 0.1 mm
% ======================================================================
% r0 = .01e-3;
% fprintf('[%s] Removing points closer than %0.1fmm\n',datetime("now"),r0*1000)
% N = numel(TARG)/3;
% 
% [TARG, keep_inds, map] = fix_locs_fast(TARG, r0);
% S = S(keep_inds);

% N_ = numel(TARG)/3;
% fprintf('[%s] %d ---> %d points\n',datetime("now"),N,N_)


% ======================================================================
% Scale model
% ======================================================================
mn = mean(TARG);
TARG = TARG-mn;
TARG = TARG./params.object.scale;
TARG = TARG+mn;



% ======================================================================
% Density compensation
% ======================================================================
voxel_size = 0.005;
fprintf('[%s] Computing scatterer density weights...\n',datetime("now"))
eps_ = .001;
w_scatter = compute_density_weights_gpu(TARG, voxel_size, eps_);
% w_scatter = ones(size(S));
 % w_scatter=1;
res.s = w_scatter.' .* S;
% res.s = S.';

% ======================================================================
% Forward Model
% ======================================================================
t_forward_model = tic;
sig = 0;

chunk_size = params.forward_model.chunk_size;
ntarg = size(TARG,1);
nchunks = floor(ntarg./chunk_size);
ntx = size(TXS,1);

w = 2i*pi.*f;
k = w./c;
forwardModelType = 'CUDA';

switch forwardModelType
    case 'MATLAB_GPU'
TARG = TARG.';

if contains(proc_type,'GPU')
    TARG = gpuArray(single(TARG));
    sigf = gpuArray(single(complex(zeros(nf,ntx,ny,nx))));
    f = gpuArray(single(f));
    c = gpuArray(single(c));
    TXS = gpuArray(single(TXS));
    k = gpuArray(single(k));
end

dist = avg_spacing_knn_gpu(TARG.', 6);
dist = dist*1000;

M = floor(nf/logging_frequency);
if M == 0
    M = 1;
end

for itx = 1:ntx
    fprintf('\n=============================================================================\n')
    fprintf('[%s] [%s] Processing transmitter %d of %d\n',datetime("now"),proc_type,itx,ntx)
    fprintf('[%s] [%s] Model size # locs: %d\n', datetime("now"),proc_type,ntarg)
    fprintf('[%s] [%s] Chunk size: %d\n',datetime("now"),proc_type,chunk_size)
    fprintf('[%s] [%s] # chunks: %d\n',datetime("now"),proc_type,nchunks)
    fprintf('[%s] [%s] Median scattering surface density: %0.2f [mm]\n',datetime("now"),proc_type,dist)
    fprintf('=============================================================================\n\n')

    TX0 = TXS(itx,:);

    for ifreq = 1:nf
        t1 = tic;
        if ifreq==1 || mod(ifreq,M)==0 || ifreq==nf
            fprintf('[%s] [%s] [TX: %d of %d] Forward Model: freq %d of %d: ', ...
                datetime("now"), proc_type, itx, ntx, ifreq, nf)
        end

        if contains(proc_type,'GPU')
            ind1 = gpuArray(single(1));
            ind2 = gpuArray(single(chunk_size));
        else
            ind1 = 1;
            ind2 = chunk_size;
        end

        for ichunk = 1:nchunks
            xtarg = TARG(1,ind1:ind2);
            ytarg = TARG(2,ind1:ind2);
            ztarg = TARG(3,ind1:ind2);
            s_loc = res.s(ind1:ind2);

            DX1 = (TX0(:,1)-xtarg).^2;
            DY1 = (TX0(:,2)-ytarg).^2;
            DZ1 = (TX0(:,3)-ztarg).^2;

            DX2 = (RX0(:,1)-xtarg).^2;
            DY2 = (RX0(:,2)-ytarg).^2;
            DZ2 = (RX0(:,3)-ztarg).^2;

            r1 = sqrt(DX1+DY1+DZ1);
            r2 = sqrt(DX2+DY2+DZ2);

            temp = sum(s_loc .* exp(-k(ifreq).*(r1+r2)) ./ (r2.*r1), 2);
            sig = temp + sig;

            ind1 = ind2+1;
            ind2 = ind1-1+chunk_size;
        end

        xtarg = TARG(1,ind1:ntarg);
        ytarg = TARG(2,ind1:ntarg);
        ztarg = TARG(3,ind1:ntarg);
        s_loc = res.s(ind1:ntarg);

        DX1 = (TX0(:,1)-xtarg).^2;
        DY1 = (TX0(:,2)-ytarg).^2;
        DZ1 = (TX0(:,3)-ztarg).^2;

        DX2 = (RX0(:,1)-xtarg).^2;
        DY2 = (RX0(:,2)-ytarg).^2;
        DZ2 = (RX0(:,3)-ztarg).^2;

        r1 = sqrt(DX1+DY1+DZ1);
        r2 = sqrt(DX2+DY2+DZ2);

        temp = sum(s_loc .* exp(-2i*pi*f(ifreq).*(r1+r2)./c) ./ (real(r2).*real(r1)), 2);
        sig = temp + sig;

        sigf(ifreq,itx,:) = sig;
        sig = sig*0;

        t2 = toc(t1);
        if ifreq==1 || mod(ifreq,M)==0 || ifreq==nf
            fprintf('%0.2f [sec]\n', t2)
        end
    end
end

t_forward_model = toc(t_forward_model);
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [sec]\n', datetime("now"),proc_type,t_forward_model)
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [min]\n', datetime("now"),proc_type,t_forward_model/60)
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [hr]\n', datetime("now"),proc_type,t_forward_model/3600)

    sigf = reshape(sigf,nf,ntx,ny,nx);
    case 'CUDA'
    res.s = single(res.s);
    TARG = single(TARG.');
    TXS = single(TXS);
    RX0 = single(RX0);
    f = single(f);
    c = single(c);
    % sigf = forward_model_mex(TARG, res.s, TXS, RX0, f, c);
    % keyboard
    TARG0 = TARG(:,1:50:end);
     TX0 = TXS(1,:);
     s = single(res.s(1:50:end));
     % keyboard
    sigf = ForwardModelCudaProject(single(TARG0), single(TXS), single(RX0), s, single(k));


end

% ======================================================================
% IMAGE RECON (unchanged)
% ======================================================================
t_recon = tic;

imgt = 0;
if contains(proc_type,'GPU')
    imgt = gpuArray(single(imgt));
end

info = [];


switch params.recon.recon_type
    case 'mf'
        nz = params.recon.mf_params.nz;
        z1 = min(TARG(3,:));
        z2 = max(TARG(3,:));
        z_padding = params.recon.mf_params.depth_pad_m; %.05;

        zs = linspace(z1-z_padding,z2+z_padding,nz);
        xpad_dist = params.recon.mf_params.width_pad_m; % this pads around the object for reconstruction
        ypad_dist = params.recon.mf_params.height_pad_m;
        xpad = floor(xpad_dist/dx);
        if rem(xpad,2)~=0
            xpad = xpad+1;
        end
        xs = 0:dx:(nx+xpad-1)*dx;
        xs = xs-mean(xs);
        nx2 = length(xs);

        ypad = floor(ypad_dist/dy);
        if rem(ypad,2)~=0
            ypad = ypad+1;
        end
        ys = 0:dy:(ny+ypad-1)*dy;
        ys = ys-mean(ys);
        ny2 = length(ys);

        [xxr,yyr] = meshgrid(xs,ys);
        zzr = zeros(size(xxr));
        xxr = xxr(:);
        yyr = yyr(:);
        zzr = zzr(:);
        X = [xxr yyr zzr];
        z0 = 0;

        if contains(proc_type,'GPU')
            X = gpuArray(single(X));
            zs = gpuArray(single(zs));
            z0 = gpuArray(single(z0));
        end

        % Resize recon image to match RA output
        % This is a bit of hack. If we switch recon to timedomain back projection we could directly match output
        % It will probably be faster recon/more efficient too.
        NY = params.recon.mf_params.img_nr_pixels_height;
        NX = params.recon.mf_params.img_nr_pixels_width;

        try
            imgf = complex(single(zeros(ntx,nz,NY,NX)));
            fprintf('[%s] [%s] Preallocating memory for image results ===> ntx,nz,ny,nx [%d x %d x %d x %d]...\n',datetime("now"),proc_type,ntx,nz,NY,NX)
            multi_tx = true;
        catch
            imgf = complex(single(zeros(nz,NY,NX)));
            fprintf('[%s] [%s] Preallocating memory for image results nz,ny,nx [%d x %d x %d x %d]...\n',datetime("now"),proc_type,nz,NY,NX)
            fprintf('[%s] [%s] Not enough memory. Storing composite result only.\n',datetime("now"),proc_type)
            multi_tx = false;
        end

        switch params.recon.mf_params.freq_window
            case "hann"
                win = hann(nf);
        end


        switch params.recon.mf_params.spatial_window
            case "tukeywin"
                winx = tukeywin(size(sigf,3));
                winy = tukeywin(size(sigf,4));
                winxy = winx*winy';
            case "none"
                winxy = ones(size(sigf));
        end



        for itx = 1:ntx
            TX0 = TXS(itx,:).*[-1 -1 1]; % hack fix, some where coords are being flipped.
            for iz = 1:nz
                if iz==1 || mod(iz,50)==0 || iz == nz
                    fprintf('[%s] [%s] [TX: %d of %d] Processing recon depth: %d of %d\n',datetime("now"),proc_type,itx,ntx,iz,nz)
                end

                targ = [z0 z0 zs(iz)];

                r1 = sqrt(sum((TX0-targ).^2,2));  % tx to target
                % r1 = reshape(r1,ny2,nx2);
                r2 = sqrt(sum((X-targ).^2,2));  % target to rx
                r2 = reshape(r2,ny2,nx2);
                imgt = imgt*0;

                for ifreq = 1:nf

                    d0 = padarray(squeeze(sigf(ifreq,itx,:,:)).*winxy,[xpad/2,ypad/2]);

                    Etx = exp(S(ifreq).*r1)./r1; % because of the sign in the exponent, this is already the conjugate
                    Erx = exp(S(ifreq).*r2)./r2;
                    G = Etx.*Erx;

                    d = fft2(d0).';
                    mf = fft2(G);


                    imgt = imgt + win(ifreq).*fftshift(ifft2(d.*mf))./max(abs(mf(:)));

                end

                if multi_tx
                    imgf(itx,iz,:,:) = gather(imresize(imgt,[NY,NX],'nearest'));
                else
                    imgf(iz,:,:) =   squeeze(imgf(iz,:,:)) + gather(abs(imresize(imgt,[NY,NX],'nearest')));

                end
            end

        end

    case 'bp'
        sigf = squeeze(sigf);


        nz = params.recon.mf_params.nz;
        z1 = min(TARG(3,:));
        z2 = max(TARG(3,:));
        x1 = min(TARG(1,:));
        x2 = max(TARG(1,:));
        y1 = min(TARG(2,:));
        y2 = max(TARG(2,:));
        xp = params.recon.bp_params.width_pad_m; %.05;
        yp = params.recon.bp_params.height_pad_m; %.05;
        zp = params.recon.bp_params.depth_pad_m; %.05;

        dx = params.recon.bp_params.spacing(1);
        dy = params.recon.bp_params.spacing(2);
        dz = params.recon.bp_params.spacing(3);

 
        % xs = linspace(x1_recon,x2_recon,nx_recon);  % width
        % ys = linspace(y1_recon,y2_recon,ny_recon);  % height
        % zs = linspace(z1_recon,z2_recon,nz_recon);  % depth

        pad_xyz = [params.recon.bp_params.width_pad_m, ...
            params.recon.bp_params.height_pad_m, ...
            params.recon.bp_params.depth_pad_m];

        step_xyz = params.recon.bp_params.spacing;   % [dx dy dz] in meters

        rcn_spc = compute_recon_bounds_from_pointcloud(TARG.', [0 0 0], [.1 .1 1]);

        xs = linspace(rcn_spc.x1, rcn_spc.x2, rcn_spc.nx);
        ys = linspace(rcn_spc.y1, rcn_spc.y2, rcn_spc.ny);
        zs = linspace(rcn_spc.z1, rcn_spc.z2, rcn_spc.nz);


        % ============================================================
% FIXED RECON EXTENTS (from JSON)
% ============================================================
if isfield(params.recon, "fixed_recon") && params.recon.fixed_recon.use_fixed == 1
    fr = params.recon.fixed_recon;
    fr.z_min = rcn_spc.z1;
    fr.z_max = rcn_spc.z1+fr.z_max;
    ct = mean(ys);
    fr.y_min=ct-fr.y_max/2;
    fr.y_max=ct+fr.y_max/2;

        
    fprintf('[%s] Using FIXED reconstruction extents from JSON\n', datetime("now"));
    fprintf('       X: [%0.3f  %0.3f]\n', fr.x_min, fr.x_max);
    fprintf('       Y: [%0.3f  %0.3f]\n', fr.y_min, fr.y_max);
    fprintf('       Z: [%0.3f  %0.3f]\n', fr.z_min, fr.z_max);
    fprintf('       voxel size: %0.4f m\n', fr.voxel_size);

    dx = fr.voxel_size;
    dy = fr.voxel_size;
    dz = fr.voxel_size;

    xs = fr.x_min : dx : fr.x_max;
    ys = fr.y_min : dy : fr.y_max;
    zs = fr.z_min : dz : fr.z_max;

else
    % ============================================================
    % ORIGINAL AUTO-BOUNDS LOGIC
    % ============================================================
    rcn_spc = compute_recon_bounds_from_pointcloud(TARG.', pad_xyz, step_xyz);

    xs = linspace(rcn_spc.x1, rcn_spc.x2, rcn_spc.nx);
    ys = linspace(rcn_spc.y1, rcn_spc.y2, rcn_spc.ny);
    zs = linspace(rcn_spc.z1, rcn_spc.z2, rcn_spc.nz);
end



        % xs = (x1-xp/2):dx:(x2+xp/2);
        % ys = (y1-yp/2):dy:(y2+yp/2);
        % zs = (z1-zp/2):dz:(z2+zp/2);

        [xx_recon,yy_recon,zz_recon] = ndgrid(xs,ys,zs);

        % [xx_recon,yy_recon,zz_recon] = ndgrid(xs,ys,zs);

        p.xx_recon = xx_recon(:);
        p.yy_recon = yy_recon(:);
        p.zz_recon = zz_recon(:);
        p.nx_recon = length(xs);
        p.ny_recon = length(ys(:));
        p.nz_recon = length(zs);
        p.npad =  params.recon.bp_params.npad;
        p.spatial_window = params.recon.bp_params.spatial_window;
        p.freq_window = params.recon.bp_params.freq_window;
        p.f1 = params.freq.f1;
        p.f2 = params.freq.f2;
        p.nf = params.freq.nf;
        p.spatial_window_param = params.recon.bp_params.spatial_window_param;
        % p.nr_elements = params.array.nr_elements;
        p.nr_elements = nr_elements;
        % [xx,yy,zz,xs0,ys0,RX0,TXS,ny,nx,dx,dy] = create_array(params);
        p.rx = RX0;


        % npad: padding amount for recon typically 5 x nf
        % spatial_window: (true|false), apply spatial window to reduce cross range
        %                 sidelobes: hardcoded ==> tukeywin(nr_elements,.95)
        % freq_window: (true|false), apply frequency window to reduce down range
        %              sidelobes: hardcoded ==> hann(nf)

        % f1: start freqeuncy (275e9Hz)
        % f2: end frequency   (305e9Hz)
        % nf: # of frequency measurements (256)

        % tx: transimtter locations [m x 3] [xpos ypos zpos]
        % rx: receiver locations [nr_elements^2 x 3]
        % nr_elements: assumes array is square, # of elements along one side
        multi_tx = false;
        if ntx>1
            multi_tx = true;
        end
        if multi_tx
            for itx = 1:ntx

                p.tx = TXS(itx,:);
                fprintf('[%s] recon tx %d of %d...\n',datetime("now"),itx,ntx)
                p.sig = squeeze(sigf(:,itx,:,:));
                % [imgf_,info] = bp_(p);
                [imgf_,info] = bp_freq_(p);
                imgf(itx,:,:,:) = gather(imgf_);

            end
            multi_tx = true;

        else
            p.tx = TXS;
            p.sig = sigf;
            % [imgf,info] = bp_(p);
            % [imgf,info] = bp_time_bistatic_(p);
            % [imgf,info] = bp_freq_(p);
            % [imgf, info, R_world2rx, rx_center] = bp_freq_rxframe_(p);
            % [imgf, info, R_world2view, scene_center] = bp_freq_viewframe_(p);


            if useCUDA

                tx = single(gather(p.tx));
                rx = single(gather(p.rx));
                sig = single(gather(p.sig));
                xx = single(gather(p.xx_recon(:)));
                yy = single(gather(p.yy_recon(:)));
                zz = single(gather(p.zz_recon(:)));
                f = single(gather(linspace(p.f1,p.f2,p.nf)));
                c = single(299792458);
                f = f(:);



                % sigf: nf x ny x nx (complex double)
                [nf, ny, nx] = size(sig);
                nXcvr = ny * nx;
              
                sig = single(reshape(sig, nf, nXcvr));   % nf x nXcvr, complex single

            
                tx_x = tx(1);
                tx_y = tx(2);
                tx_z = tx(3);
                rx_x = rx(:,1);
                rx_y = rx(:,2);
                rx_z = rx(:,3);
                nVox = length(xx);
                nXCVR = numel(rx);
                % imgf = bp_freq_bistatic_cuda(tx, rx, sig, xx, yy, zz, f, c);
                imgf = bp_freq_bistatic_cuda( ...
                    tx_x, tx_y, tx_z, ...
                    rx_x, rx_y, rx_z, ...
                    sig, xx, yy, zz, f, c);
                % [imgf_dbg, info] = bp_freq_bistatic_(p);
                imgf = reshape(imgf, [numel(xs), numel(ys), numel(zs)]);

                info = [];
            else
                [imgf, info] = bp_freq_bistatic_(p);

            end


        end

end

t_recon = toc(t_recon);

t_save_data = tic;

TARG_ = TARG.';
if contains(proc_type,'GPU')
    TARG = gather(TARG);
    TARG_ = gather(TARG_);
    TX0 = gather(TX0);
    RX0 = gather(RX0);
end




%================================================================================
%% SAVE DATA?
%================================================================================
if params.options.saveData

    sigf = gather(sigf);
    array.xx = xx;
    array.yy = yy;
    array.xs0 = xs0;
    array.ys0 = ys0;
    array.RX0 = RX0;
    array.TXS = TXS;
    array.nx = nx;
    array.ny = yy;
    array.dx = dx;
    array.dy = dy;

    sn = fullfile(output_folder,'fm.mat');
    fprintf('[%s] Saving data: %s\n',datetime("now"),sn)
    try
        save(sn,'sigf','params','f','array','-v7')
    catch ME
        fprintf('[%s] error saving %s\n',datetime("now"),sn)
        rethrow(ME)
    end

    sn = fullfile(output_folder,'recon.mat');
    imgf = gather(imgf);
    fprintf('[%s] Saving data: %s\n',datetime("now"),sn)
    try
        save(sn,'imgf','info','params','xs','ys','zs','-v7')
    catch ME
        fprintf('[%s] error saving %s\n',datetime("now"),sn)
        rethrow(ME)
    end

    if params.options.saveModel
        sn = fullfile(output_folder,'model.mat');
        fprintf('[%s] Saving data: %s\n',datetime("now"),sn)
        save(sn,'TARG','params')
    end


end



t_save_data = toc(t_save_data);
t_create_plots_and_save = tic;
%+===============================================================================
%% PLOT RECON SLICES
%+===============================================================================
% first plot coherent additions
clims = [-45 0];
fh = figure('visible','off');
subplot(1,3,1)
if multi_tx
    imagesc(-xs,ys,20*log10(imnorm(squeeze(max(abs(sum(abs(imgf))),[],4)))).'),axis image, colormap jet,axis xy
    title(sprintf('Max Projection - XY\nSum Across TXs'))
else
    imagesc(-xs,ys,20*log10(imnorm(squeeze(max(abs(imgf),[],3)).'))),axis image, colormap jet,axis xy
    title('Max Projection - XY')
end

xlabel('Width [m]')
ylabel('Depth [m]')
set(gca,"fontsize",12)
clim(clims)
colorbar

subplot(1,3,2)
if multi_tx
    imagesc(-xs,zs,20*log10(imnorm(squeeze(max(abs(sum(abs(imgf))),[],2))))),axis image, colormap jet,axis ij
    title(sprintf('Max Projection - XZ\nSum Across Txs'))
else
    imagesc(-xs,zs,20*log10(imnorm(squeeze(max(abs(imgf),[],2)).'))),axis image, colormap jet,axis xy
    title('Max Projection - XZ')
end
xlabel('Width [m]')
ylabel('Height [m]')
set(gca,"fontsize",12)
colorbar
clim(clims)

subplot(1,3,3)
if multi_tx
    imagesc(ys,zs,20*log10(imnorm(squeeze(max(abs(sum((imgf))),[],3))))'),axis image, colormap jet,axis xy
    title(sprintf('Max Projction - ZX\nSum Across Txs'))
else
    imagesc(ys,zs,20*log10(imnorm(squeeze(max(abs(imgf),[],1))))'),axis image, colormap jet,axis xy
    title('Max Projction - YZ')
end
xlabel('Depth [m]')
ylabel('Height [m]')
set(gca,"fontsize",12)
set(gcf,'color','w')

colorbar
clim(clims)

fn = fullfile(output_folder,'max_projections.png');
fprintf('[%s] Saving figure: %s\n',datetime("now"),fn)
exportgraphics(fh,fn);
close(fh)



fh = figure('Visible','off');
try
    pcshow(gather(real(RX0)),'m'), hold on
    pcshow(gather(real(TXS)),'r')
    colormap jet
    set(gcf,'color', [.2 .2 .2])
catch err
    err
    fprintf('\n[%s] **** UNABLE TO RENDER POINTCLOUD DATA **** ===> USING PLOT3\n\n',datetime("now"))
    plot3(real(RX0(:,1 )), real(RX0(:,2)), real(RX0(:,3)),'.m'), hold on
    plot3(real(TXS(:,1)), real(TXS(:,2)), real(TXS(:,3)),'sr')

end
% plot3(real(TXS(:,1)),real(TXS(:,3)),real(TXS(:,2)),'sr')
axis image

try
    pcshow(TARG_(:,[1 2 3]),res.s)
catch err
    err
    fprintf('\n[%s] **** UNABLE TO RENDER POINTCLOUD DATA **** ===> USING PLOT3\n\n',datetime("now"))
    plot3(TARG_(1:10:end,1),TARG_(1:10:end,2),TARG_(1:10:end,3),'.c')
    axis image
end
colorbar
colormap jet
set(gcf,'color', [.2 .2 .2])
fn = fullfile(output_folder,'model.jpg');
fprintf('[%s] Saving figure: %s\n',datetime("now"),fn)
exportgraphics(fh,fn,'Resolution',300)

fInd = nf;
itx = 1;
fh = figure('Visible','off');
subplot(1,2,1)
imagesc(xs0,ys0,squeeze(abs(sigf(fInd,:,:)))),axis image, colormap jet
set(gca,'fontsize',18)
xlabel('x [m]')
ylabel('y [m]')
title(sprintf('Measured Data\nMag @ %0.1fGHz',f(fInd)/1e9))

subplot(1,2,2)
imagesc(xs0,ys0,squeeze(angle(sigf(fInd,:,:)))),axis image, colormap jet
set(gca,'fontsize',18)
xlabel('x [m]')
ylabel('y [m]')

title(sprintf('Measured Data\nPhase @ %0.1fGHz',f(fInd)/1e9))
set(gcf,'color','w')
fn = fullfile(output_folder,'fields.png');
fprintf('[%s] Saving figure: %s\n',datetime("now"),fn)
exportgraphics(fh,fn)
close(fh)

t_create_plots_and_save = toc(t_create_plots_and_save);
t_total_process_time = toc(T0);

timing.load_data = t_load_model;
timing.forward_model = t_forward_model;
timing.recon = t_recon;
timing.save_data = t_save_data;
timing.create_plots_and_save = t_create_plots_and_save;
timing.total = t_total_process_time;
timing.recon_info = info;

fn = fullfile(output_folder,'timing.mat');
fprintf('[%s] saving timing info: %s\n',datetime("now"),fn);
save(fn,'timing')


sys = get_system_info();


fn = fullfile(output_folder,'sys.mat');
fprintf('[%s] saving system info: %s\n',datetime("now"),fn);
save(fn,'sys')

fprintf('\n\n=============================================================================\n')
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [sec]\n', datetime("now"),proc_type,t_forward_model)
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [min]\n', datetime("now"),proc_type,t_forward_model/60)
fprintf('[%s] [%s] Forward Model Processing Time: %0.1f [hr]\n\n', datetime("now"),proc_type,t_forward_model/3600)

fprintf('[%s] [%s] Recon Processing Time: %0.1f [sec]\n', datetime("now"),proc_type,t_recon)
fprintf('[%s] [%s] Recon Processing Time: %0.1f [min]\n', datetime("now"),proc_type,t_recon/60)
fprintf('[%s] [%s] Recon Processing Time: %0.1f [hr]\n\n', datetime("now"),proc_type,t_recon/3600)

fprintf('=============================================================================\n')
fprintf('[%s] [%s] Processing transmitter %d of %d\n',datetime("now"),proc_type,itx,ntx)
fprintf('[%s] [%s] Model size # locs: %d\n', datetime("now"),proc_type,ntarg)
fprintf('[%s] [%s] Chunk size: %d\n',datetime("now"),proc_type,chunk_size)
fprintf('[%s] [%s] # chunks: %d\n',datetime("now"),proc_type,nchunks)
% fprintf('[%s] [%s] Median scattering surface density: %0.2f [mm]\n',datetime("now"),proc_type,dist)
fprintf('=============================================================================\n\n')

fprintf('=============================================================================\n')
fprintf('[%s] [%s] Total process time: %0.1f [min]\n',datetime("now"),proc_type,t_total_process_time/60)
fprintf('=============================================================================\n')

fprintf('OUTPUT FOLDER: %s\n',output_folder)
%================================================================================
%% STOP COMMAND LINE LOGGING
%================================================================================
try
    diary off;
catch
end

return

end




%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%================================================================================================
% PRIVATE HELPER FUNCTIONS
%================================================================================================
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function recon = compute_recon_bounds_from_pointcloud(TARG, pad_xyz, step_xyz)
% COMPUTE_RECON_BOUNDS_FROM_POINTCLOUD
%   Compute padded reconstruction-space extents and grid sizes
%   from a 3D point cloud.
%
% INPUTS:
%   TARG     : [N x 3] point cloud (x,y,z) in meters
%   pad_xyz  : [1 x 3] padding amounts [pad_x, pad_y, pad_z] in meters
%   step_xyz : [1 x 3] voxel spacing [dx, dy, dz] in meters
%
% OUTPUT:
%   recon struct with fields:
%       .x1, .x2
%       .y1, .y2
%       .z1, .z2
%       .nx, .ny, .nz
%       .dx, .dy, .dz
%
% NOTES:
%   - Padding is symmetric.
%   - Grid size is computed from physical extent / step size.
%   - Ensures nx,ny,nz are integers and include both endpoints.
%
% arguments
%     TARG (:,3) double
%     pad_xyz (1,3) double {mustBeNonnegative}
%     step_xyz (1,3) double {mustBePositive}
% end

% Extract padding
pad_x = pad_xyz(1);
pad_y = pad_xyz(2);
pad_z = pad_xyz(3);

% Extract step sizes
dx = step_xyz(1);
dy = step_xyz(2);
dz = step_xyz(3);

% Compute bounding box
xmin = min(TARG(:,1));
xmax = max(TARG(:,1));

ymin = min(TARG(:,2));
ymax = max(TARG(:,2));

zmin = min(TARG(:,3));
zmax = max(TARG(:,3));

% Apply symmetric padding
recon.x1 = xmin - pad_x;
recon.x2 = xmax + pad_x;

recon.y1 = ymin - pad_y;
recon.y2 = ymax + pad_y;

recon.z1 = zmin - pad_z;
recon.z2 = zmax + pad_z;

% Compute grid sizes
recon.dx = dx;
recon.dy = dy;
recon.dz = dz;

recon.nx = ceil((recon.x2 - recon.x1) / dx) + 1;
recon.ny = ceil((recon.y2 - recon.y1) / dy) + 1;
recon.nz = ceil((recon.z2 - recon.z1) / dz) + 1;

fprintf('[%s] Recon bounds computed from point cloud:\n', datetime("now"));
fprintf('    X: [%0.3f  %0.3f] m   nx = %d\n', recon.x1, recon.x2, recon.nx);
fprintf('    Y: [%0.3f  %0.3f] m   ny = %d\n', recon.y1, recon.y2, recon.ny);
fprintf('    Z: [%0.3f  %0.3f] m   nz = %d\n', recon.z1, recon.z2, recon.nz);
end




%================================================================================================
% Load parameter json file
%================================================================================================
function params = load_params(param_file)
fprintf('[%s] Loading system parameters from: %s\n', datetime("now"), param_file);
fid = fopen(param_file,'r');
str = textscan(fid,'%c','Whitespace','');
str = str{1}';
fclose(fid);
params = jsondecode(str);
end


%================================================================================================
% Check/verify processing type (GPU/CPU)
%================================================================================================
function proc_type = check_proc_type(params)

proc_type = "CPU";
gpuAvailable = canUseGPU;
if gpuAvailable && params.options.useGPU == 1
    proc_type = "GPU";
end

%===================================================================================
% Check if GPU is AVAILABLE
%===================================================================================
if gpuAvailable & params.options.useGPU == 1
    fprintf("\n[%s] **** GPU AVAILABLE **** ===> RESETTING GPU\n\n",datetime("now"))
    gpuDevice(1);
end

if ~gpuAvailable
    fprintf("\n[%s] **** NO GPU AVAILABLE **** ===> USING CPU\n\n",datetime("now"))
end

if params.options.useGPU == 0
    fprintf("\n[%s] **** CPU PROCESSING SELECTED ****\n\n",datetime("now"))
end

end


%================================================================================================
% Create array and transmitters
%================================================================================================
function [xx,yy,zz,xs0,ys0,RX0,TXS,ny,nx,dx,dy] = create_array(params)

dx = params.array.dx;                    	            % 5.169e-4m mean element pitch from specifications
dy = dx;                          						% pitch is the same on both directions
nr_elements = params.array.nr_elements;                 % 131x131 from specifications
xs0 = 0:dx:dx*((nr_elements-1)*params.array.scale_factor);

xs0 = xs0-mean(xs0);
nx = length(xs0);
ys0 = 0:dy:dy*((nr_elements-1)*params.array.scale_factor);
ys0 = ys0-mean(ys0);
ny = length(ys0);
xa = xs0(end);
ya = xa;

[xx,yy] = meshgrid(xs0,ys0);
zz = zeros(size(xx));
xx = xx(:);
yy = yy(:);
zz = zz(:);
RX0 = [xx yy zz-0i];


TX_X = params.array.tx.x;
TX_Y = params.array.tx.y;
TX_Z = params.array.tx.z;

TX_X = TX_X(:);
TX_Y = TX_Y(:);
TX_Z = TX_Z(:);

TXS = [TX_X TX_Y TX_Z];

end