%--------------------------------------------------------------------------
% Created for: Matlab Version 7.2 (R2006a)
% Uses:
% Revisions:
%--------------------------------------------------------------------------
% function res = sg_filt(data,filter_length,filter_order,derivative_order,N);
%
% Polynomial filtering method of Savitsky-Golay. sg_sm is essentially a weighted
% moving average which fits a polynomial to the data in a least square
% sense. Smoothing as well as derivatives can be perfomed with this
% algorithm.
%
%--------------------------------------------------------------------------
% inputs:  
%    
%           data = vector of signals to be filtered
%           *** the derivative is calculated for each ROW **
%
%           filter_length = filter length, must be an odd number
%
%           filter_order = filter order (1 = linear, 2 = quadratic filter,
%            4 = quartic)
%
%           derivative_order = derivative order (0 = smoothing, 1 = first
%           derivative, 2 = second derivative . .
%           
%           N = number of iterations, default = 1
%
% outputs:
%
%             res: filtered data
%--------------------------------------------------------------------------
% for:        
% other info: None
%--------------------------------------------------------------------------
%   Modification log (Date, Name, Description of modifications)
%--------------------------------------------------------------------------

function res = sg_filt(data,filter_length,filter_order,derivative_order,N)
data= data(:)';
flag = 0;
if ~exist('N','var')
    N = 1;
end
if mod(filter_length,2) == 0
    ftemp = filter_length;
     filter_length = filter_length -1;
     display(['function sg_sm error: filter length ' num2str(ftemp) ...
         ' is not an odd integer. filter length changed to : ' num2str(filter_length)]);
end

dorder = derivative_order;
forder = filter_order;
len = filter_length;

d_left = data(1:len);
d_left = d_left(end:-1:1);

d_right = data(end-len+1:end);
d_right = d_right(end:-1:1);
data = [ones(1,2*len)*mean(d_left) data ones(1,2*len)*mean(d_right)];

% ********************************
if size(data,2) > 1
    data = data';
    flag = 1;
end
% ***********************************
[m,n] = size(data);
dorder = dorder + 1;
%==========================================================================
% *** check inputs ***
%==========================================================================
if rem(len,2)-1 ~= 0
 error('filter length is not an odd integer')
elseif (forder) < (dorder)
 display(['function sg_sm error: derivative order ' num2str(dorder) ...
         ' is not less than filter order' num2str(forder)]);
end

for ii = 1:N
%==========================================================================
% *** calculate filter coefficients ***
%==========================================================================
lc = (len-1)/2;                                 % index
X = [-lc:lc]'*ones(1,forder+1);
p = ones(len,1)*[0:forder];                     % polynomial terms
X = X.^p;                                       % polynomial coefficients
F = pinv(X);                                    % invert

%==========================================================================
% *** filter via convolution and take care of the end points ***
%==========================================================================
res = conv(data,F(dorder,:));

res = res(lc+1:length(res)-lc);
res(1:lc) = res(lc+1);
res(m-lc:m) = res(m-lc-1);

data = res;

end

if flag == 1
    res = res';
end

res = res(2*len+1:end-2*len);
