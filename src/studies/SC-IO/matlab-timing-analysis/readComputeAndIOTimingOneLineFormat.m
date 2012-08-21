function [ proc_events ] = readComputeAndIOTimingOneLineFormat( filename, proc_type )
%readComputeAndIOTiming reads the timing file
%   format of timing file is :
%       pid, hostname, filename, stagename, stagetype, stagename, stagetype, ...
%       next line: values
%   then repeat.
    proc_events = {};

    fid = fopen(filename);
    
    % iterate through the lines
    tline = fgetl(fid);
    if ischar(tline) && (strcmp(tline, 'v2.1') == 1)
        % processing version 2.1 (has annotation output, which is just the
        % size of the data being outputted. also has group information. ignore for now.
        tline = fgetl(fid);  % skip header
        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %*d %*s %s', 1, 'delimiter', ',');
            if ~strcmp(temp1{3}, proc_type) || strcmp(proc_type, '*')
                temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64 %u64', 'delimiter', ',', 'emptyvalue', 0);

                proc_events = [proc_events; temp1, temp2];
                clear temp2;
            else
                fprintf(1, 'SKIP %s lines %s\n', proc_type, cell2mat(temp1{3}));
            end
            clear temp1;
            tline = fgetl(fid);
        end
    elseif ischar(tline) && (strcmp(tline, 'v2') == 1)
        % processing version 2 (has annotation output, which is just the
        % size of the data being outputted.
        tline = fgetl(fid);  % skip header
        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            if strcmp(temp1{3}, 'w')
                temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64 %u64', 'delimiter', ',', 'emptyvalue', 0);

                proc_events = [proc_events; temp1, temp2];
                clear temp2;
            else
                fprintf(1, 'SKIP non worker lines %s\n', cell2mat(temp1{3}));
            end
            clear temp1;
            tline = fgetl(fid);
        end        
    else
        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            if strcmp(temp1{3}, 'w')
                temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64', 'delimiter', ',');

                proc_events = [proc_events; temp1, temp2];
                clear temp2;
            else
                fprintf(1, 'SKIP non worker lines %s\n', cell2mat(temp1{3}));
            end
            clear temp1;
            tline = fgetl(fid);
        end
    end
    clear tline;
    fclose(fid);
    
    %% merge action and comm on the same node
    
    nodes = unique([proc_events{:, 1}]);
    pes2 = cell(size(nodes, 2), size(proc_events, 2));
    
    for j = 1:size(nodes, 2)
        pos = find([proc_events{:, 1}] == nodes(j));
        pe2 = proc_events(pos, :);
        pes2{j, 1} = nodes(j);
        pes2{j, 2} = pe2{1, 2};
        pes2{j, 3} = pe2{1, 3};
        for i = 4:size(proc_events, 2)
            pes2{j, i} = cat(1, pe2{:, i});
        end
        
    end
    proc_events = pes2;
end

 