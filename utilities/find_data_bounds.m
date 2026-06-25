function [ind1,ind2,z0,zs,data] = find_data_bounds(data,cut_off,offsets)

if nargin<2
    cut_off = .05;
    offsets = [20 75];
end

if nargin<3
    offsets = [20 75];
end
% finds bounds around a person

c=299792458;
bw = 305e9-275e9;
dr = c./(2*bw);

imgn = imnorm(abs(data));
imgn = abs(imgn-median(imgn,2));
indx = find(imgn>cut_off,50,'first');
[iz,~,~]=ind2sub(size(data),indx);
ind1 = min(iz)-offsets(1);
ind2 = ind1+offsets(2);



data = data(ind1:ind2,:,:);
z0 = min(iz)*dr;
zs = 0:dr:dr*(ind2-ind1);


