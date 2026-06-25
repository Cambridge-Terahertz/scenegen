function tar_and_zip(srcFolder, outputName, matFileToExclude, extExcludeList)

    if nargin < 4
        error('Usage: tar_and_zip(srcFolder, outputName, matFileToExclude, extExcludeList)');
    end

    srcFolder = string(srcFolder);
    outputName = string(outputName);
    matFileToExclude = string(matFileToExclude);
    extExcludeList = string(extExcludeList);

    % Normalize root folder
    srcFolder = char(srcFolder);
    rootLen = length(srcFolder) + 2; % for stripping "srcFolder\" prefix

    % ---------------------------------------------------------------------
    % 1. Collect all files recursively (relative paths)
    % ---------------------------------------------------------------------
    allFiles = dir(fullfile(srcFolder, '**', '*'));
    allFiles = allFiles(~[allFiles.isdir]);

    filesToInclude = {};

    for k = 1:numel(allFiles)
        absPath = fullfile(allFiles(k).folder, allFiles(k).name);

        % Skip folders "~" and "hideme"
        parts = split(string(allFiles(k).folder), filesep);
        if any(parts == "~") || any(parts == "hideme")
            continue
        end

        % Skip specific .mat file
        if allFiles(k).name == matFileToExclude
            continue
        end

        % Skip excluded extensions
        skip = false;
        for ext = extExcludeList
            if endsWith(allFiles(k).name, extractAfter(ext, '*'))
                skip = true;
                break
            end
        end
        if skip
            continue
        end

        % Convert absolute → relative path
        relPath = absPath(rootLen:end);

        filesToInclude{end+1} = relPath; %#ok<AGROW>
    end

    % ---------------------------------------------------------------------
    % 2. Create TAR archive
    % ---------------------------------------------------------------------
    tarFile = outputName + ".tar";
    fprintf("Creating TAR: %s\n", tarFile);
    tar(tarFile, filesToInclude, srcFolder);

    % ---------------------------------------------------------------------
    % 3. Create ZIP archive
    % ---------------------------------------------------------------------
    zipFile = outputName + ".zip";
    fprintf("Creating ZIP: %s\n", zipFile);
    zip(zipFile, tarFile);

    fprintf("Done. Created:\n  %s\n  %s\n", tarFile, zipFile);
end
