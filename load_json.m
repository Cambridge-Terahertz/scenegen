function json = load_json(filepath, filename)

if nargin < 2
    [filename,filepath]=uigetfile('*.json');
end


fid = fopen(fullfile(filepath,filename),'r');
str = textscan(fid,'%c');
fclose(fid);
str = str{1}';
try
    json = jsondecode(str);

catch err

    err_str = err.message;
    col = regexp(err_str, '(?<=column\s+)\d+', 'match');
    n = str2double(col{1});
    ind1 = n-10;
    M = length(string);
    if ind1 < 1
       ind1 = 1;
    end

    ind2 = n+10;
    if ind2>length(str)
        ind2 = M;
    end
    fprintf('[%s] error @  position: %d\n',datetime("now"),n)
    fprintf('\t %s ---> %s <---- %s\n',str(ind1:n-1), str(n), str(n+1:ind2))

end