function [pos1,pos2, im1,im2] = find_first_refelction(data,cut_off)
data = imnorm(abs(data));
[nz,ny,nx] = size(data);
data = reshape(data,nz,[]);
pos1 = zeros(ny*nx,1);
pos2 = pos1;
im1 = pos1;
im2 = pos1;
for ii = 1:size(data,2)
    ray = data(:,ii);
    temp = find(ray>cut_off,1,'first');
    if isempty(temp)
        temp = 0;
    end
    pos1(ii)=temp;
    p1 = temp;
    temp = find(ray>cut_off,1,'last');
      % temp = find(ray(p1:p1+50)<cut_off,1,'first');
      % temp = temp+p1;
    if isempty(temp)
        temp = 0;
    end
   
    pos2(ii)=temp;
    ind1 = pos1(ii);
    ind2 = pos2(ii);
    if any([ind1 ind2]==0)
        % skip
    else
     ray = ray(pos1(ii):pos2(ii));
    im1(ii) = max(ray);
    im2(ii) = mean(ray);
    end
end

pos1 = reshape(pos1,ny,nx);
pos2 = reshape(pos2,ny,nx);
im1 = reshape(im1,ny,nx);
im2 = reshape(im2,ny,nx);