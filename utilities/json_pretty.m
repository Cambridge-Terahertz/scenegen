function txt = json_pretty(obj, indent)
    if nargin < 2
        indent = 0;
    end
    sp = repmat(' ',1,indent);

    if isstruct(obj)
        fields = fieldnames(obj);
        parts = {};
        for i = 1:numel(fields)
            key = fields{i};
            val = json_pretty(obj.(key), indent+4);
            parts{end+1} = sprintf('%s"%s": %s', sp, key, val);
        end
        txt = sprintf('{\n%s\n%s}', strjoin(parts, ",\n"), sp);

    elseif iscell(obj)
        parts = {};
        for i = 1:numel(obj)
            parts{end+1} = json_pretty(obj{i}, indent+4);
        end
        txt = sprintf('[\n%s\n%s]', strjoin(parts, ",\n"), sp);

    elseif isnumeric(obj)
        if isscalar(obj)
            txt = sprintf('%g', obj);
        else
            txt = sprintf('[%s]', strjoin(arrayfun(@(x) sprintf('%g',x), obj, 'UniformOutput', false), ', '));
        end

    elseif ischar(obj)
        txt = sprintf('"%s"', obj);

    else
        txt = sprintf('"%s"', string(obj));
    end
end
