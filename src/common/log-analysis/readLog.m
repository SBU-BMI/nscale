function [ events names ] = readLog( dirname, fstruct )
%readLog reads the timing file
%   format of timing file is :
%       pid val, hostname val, filename val, stagename val, stagetype val, stagename val, stagetype val, ...
%       next line: values
%   then repeat.
%   return a cell array: each row is 1 MPI process, each cell in row is one
%   field
%   TODO:  update to use struct for the row payloads

    events = cell(0, 8);
    filename = fullfile(dirname, fstruct.name);

    if (fstruct.bytes == 0)
        fprintf(2, 'file %s is zero byte\n', filename);
        return;  
    end

    % open and read the file into memory
    fid = fopen(filename);    
    if (fid == -1)
        fprintf(2, 'cannot open file %s\n', filename);
        return;
    end
    
    %fgetl is slow.  on a 20MB file with multiple lines to exlude,
    %this takes a long time (readPerf function code below is an example of speed,
    %without the additonal processing.)
    t = textscan(fid, '%s', 'Delimiter', '\n', 'MultipleDelimsAsOne', 1, 'ReturnOnError', 0, 'bufsize', fstruct.bytes + 1);
    r = strtrim(t{1});  % remove leading and trailing spaces
    clear t;
    
    r( cellfun(@isempty,r), : ) = [];  % remove blank lines
    r( strncmp('#', r, 1), :) = [];  % remove comment liness
    
    fclose(fid);
    
    % check number of lines
    numlines = size(r, 1);
    if (numlines == 0)  
        fprintf(2, 'file %s contains whitespace and comments only\n', filename);
        return;
    end
    
    % now we can also prealloc events.  - HUGE speed up.

    % iterate through the lines

    % parameters
    procid_pattern = '';
    procval_pattern = '';
    delimiter = ',';
%    event_delimiter = ';';
    
    %% check file version and set the appropriate variables.
    % check header for version
    tline = r{1};

    if (strncmp(tline, 'v3.0', 4) == 1)
        % TODO: version 3.0?
        fprintf(2, 'unsupported future log format');
        return;
    elseif (strncmp(tline, 'v2.2', 4) == 1) 
    	% TODO: ver 2.2 should extend 2.1, with an additional field of message target.
        fprintf(2, 'unsupported future log format');
        return;
    elseif (strncmp(tline, 'v2.1', 4) == 1)
        % processing version 2.1 (has annotation output, which is just the
        % size of the data being outputted. also has group information. ignore for now.
        names = struct('pid', 1, 'hostName', 2, 'group', 9, 'sessionName', 3, 'eventName', 4, 'eventType', 5, 'startT', 6, 'endT', 7, 'attribute', 8);
        procid_pattern = '%*s %d %*s %s %*s %d %*s %s';
        procval_pattern = '%s %d %u64 %u64 %u64';
    elseif ischar(tline) && (strncmp(tline, 'v2', 2) == 1)
        names = struct('pid', 1, 'hostName', 2, 'sessionName', 3, 'eventName', 4, 'eventType', 5, 'startT', 6, 'endT', 7, 'attribute', 8);
        procid_pattern = '%*s %d %*s %s %*s %s';
        procval_pattern = '%s %d %u64 %u64 %u64';
    else
        names = struct('pid', 1, 'hostName', 2, 'sessionName', 3, 'eventName', 4, 'eventType', 5, 'startT', 6, 'endT', 7);
        procid_pattern = '%*s %d %*s %s %*s %s';
        procval_pattern = '%s %d %u64 %u64';
    end
    
    lines = r(strncmp('pid', r, 3), :);
    if (size(lines, 1) == 0)
        fprintf(2, '%s is not a log output file\n', filename);
        return;
    end
    clear events;
    events = cell(size(lines, 1), size(fieldnames(names), 1));
    
    fns = fieldnames(names);
    positions = zeros(size(fns, 1), 1);
    
    for i = 1:size(fns,1)
       positions(i) = names.(fns{i});
    end
    
    % get the process information
    idx = 1;
    for line = lines'
        tline = line{1};
        [temp1 pos] = textscan(tline, procid_pattern, 1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', 0, 'ReturnOnError', 0);

        if (pos < length(tline))
            % if there is process event values, parse them.
            temp2 = textscan(tline(pos+1:end), procval_pattern, 'Delimiter', delimiter, 'emptyvalue', 0, 'MultipleDelimsAsOne', 0, 'ReturnOnError', 0);
        else 
            temp2 = cell(1, size(events,2) - size(temp1,2));
        end
        events(idx, positions) = [temp1 temp2];
        idx = idx+1;
    end
    clear temp1;
    clear temp2;
    
    clear r;
    
    
    function readPerf
        fn = 'TCGA.separate.jaguar.p10240.f100000.MPI_AMR.b4.io7680-16.is7680-1.csv'
        dir = '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1'

        % 0.61 sec on the input.
        tic;
        fid = fopen([dir '/' fn]);
        r = textscan(fid, '%s', 'Delimiter', '\n', 'bufsize', 21000000);
        fclose(fid);
        toc
        
        % 0.72 sec on the input
        tic;
        fid = fopen([dir '/' fn]);
        str = char(fread(fid)');
        fclose(fid);
        r = textscan(str, '%s', 'Delimiter', '\n', 'bufsize', 21000000);
        toc

        %3.55 sec on the input
        tic;
        sizB = 50000;
        sizS = 50000;
        s = cell(sizS, 1);

        fid = fopen([dir '/' fn]);
        lineCt = 1;
        tline = fgetl(fid);
        while ischar(tline)
           s{lineCt} = tline;
           lineCt = lineCt + 1;
           % grow s if necessary
           if lineCt > sizS
               s = [s;cell(sizB,1)];
               sizS = sizS + sizB;
           end
           tline = fgetl(fid);
        end
        % remove empty entries in s
        s(lineCt:end) = [];
        fclose(fid);
        toc
    end
    
    


end

 
