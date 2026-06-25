function data = imnorm(data)

data = data - min(data(:));
data = data./max(data(:));


