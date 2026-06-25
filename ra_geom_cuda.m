function [xx,yy,zz,xs0,ys0,RX0,TXS,ny,nx,dx,dy,nr_elements] = ra_geom_cuda(params)


doQC = false;

switch params.array.array_type
    case 'exact'

        pitch = 508e-6; % element pitch
        dx = pitch;
        dy = dx;
        sep = 518e-6;   % tile separation
        ntiles = 13*params.array.scale_factor;
     
        element_nr = 10;

        ny = ntiles*element_nr;
        nx = ntiles*element_nr;
        
        nr_elements = (element_nr*ntiles);
        xs0 = 0:pitch:pitch*(element_nr-1);
        xs0 = xs0-mean(xs0);
        ys0 = xs0;
        w = xs0(end)-xs0(1);
        h = ys0(end)-ys0(1);
        [xx,yy] = ndgrid(xs0,ys0);

        tile_ctrs = 0:w+sep:(w+sep)*(ntiles-1);
        [xgrid, ygrid] = ndgrid(tile_ctrs,tile_ctrs);

        xa = xgrid(:)+xx(:).';
        ya = ygrid(:)+yy(:).';
        X = xa+1i.*ya;
        X = unique(X);
        xx = real(X(:));
        yy = imag(X(:));
        zz = zeros(size(xx));

        if doQC
            figure
            plot(xa,ya,'.')
            hold on
            plot(xgrid,ygrid,'sk')
            axis image

            bx = w*[-1 1 1 -1 -1]/2;
            by = h*[-1 -1 1 1 -1]/2;
            xgrid = xgrid(:);
            ygrid = ygrid(:);
            for ii = 1:13*13
                plot(bx+xgrid(ii),by+ygrid(ii),'k')
            end
            xlabel('Distance, m')
            ylabel('Distance, m')
            title('Reflect Array Geometry - Exact')
            set(gca,'fontsize',18)
            set(gcf,'color','w')
        end

    case 'regular' % regular grid

        dx = 5.169e-4; %params.array.dx;                    	            % 5.169e-4m mean element pitch from specifications
        dy = dx;                          						% pitch is the same on both directions
        nr_elements = 130;                 % 131x131 from specifications
        xs0 = 0:dx:dx*((nr_elements-1)*params.array.scale_factor);

        xs0 = xs0-mean(xs0);
        nx = length(xs0);
        ys0 = 0:dy:dy*((nr_elements-1)*params.array.scale_factor);
        ys0 = ys0-mean(ys0);
        ny = length(ys0);
        xa = xs0(end);
        ya = xa;

        % [xx,yy] = ndgrid(xs0,ys0);
        % zz = zeros(size(xx));
        % xx = xx(:);
        % yy = yy(:);
        % zz = zz(:);
        [xx,yy] = meshgrid(xs0, ys0);   % IMPORTANT: meshgrid, not ndgrid
        zz = zeros(size(xx));

        RX0 = [xx(:), yy(:), zz(:)];    % x, y, z order
         % RX0 = [xx(:), zz(:), yy(:)];    % x, y, z order
end


% RX0 = [xx zz yy-0i];


TX_X = params.array.tx.x;
TX_Y = params.array.tx.y;
TX_Z = params.array.tx.z;

TX_X = TX_X(:);
TX_Y = TX_Y(:);
TX_Z = TX_Z(:);

TXS = [TX_X TX_Y TX_Z];
