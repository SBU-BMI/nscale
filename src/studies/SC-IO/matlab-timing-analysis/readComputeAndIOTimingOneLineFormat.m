function [ proc_events ] = readComputeAndIOTimingOneLineFormat( dirname, fstruct, proc_type, event_names )
%readComputeAndIOTiming reads the timing file
%   format of timing file is :
%       pid, hostname, filename, stagename, stagetype, stagename, stagetype, ...
%       next line: values
%   then repeat.

    tic;
    filename = fullfile(dirname, fstruct.name);
    fprintf(1, 'reading %s\n', filename);
    fid = fopen(filename);
    
    %fgetl is slow.  on a 20MB file with multiple lines to exlude,
    %this takes a long time (readPerf function code below is an example of speed,
    %without the additonal processing.)
    r = textscan(fid, '%s', 'Delimiter', '\n', 'bufsize', fstruct.bytes + 1);
    r = r{1};
    fclose(fid);
    toc
    
    numlines = size(r, 1);
    
    % now we can also prealloc proc_events.  - HUGE speed up.
    tic;
    fprintf(1, 'parsing the file content \n');
    % iterate through the lines
    linenum = 1;
    outlinenum = 1;
    % skip header
    if (linenum < numlines)
        tline = r{linenum};
    else
        tline = '';
    end
    if ischar(tline) && (strcmp(tline, 'v2.1') == 1)
        proc_events = cell(size(r, 1) - 1, 8);
   
        
        % processing version 2.1 (has annotation output, which is just the
        % size of the data being outputted. also has group information. ignore for now.
        linenum = linenum + 1;
        if (linenum < numlines)
            tline = r{linenum}; 
        else
            tline = '';
        end
        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %*d %*s %s', 1, 'delimiter', ',');
            if (isempty(find(strcmp(temp1{3}, proc_type), 1)) || ~isempty(find(strcmp('*', proc_type), 1))) 
                if (length(tline(pos+1:end)) > 0)
                    temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64 %u64', 'delimiter', ',', 'emptyvalue', 0);

                    % TODO: NEED TO PUT IN THE MECHANISM TO FILTER OUT SOME
                    % EVENTS BY NAME
                    for i=1:size(event_names, 2)
                        idx = ~strcmp(event_names{i}, temp2{1});
                        for j = 1:size(temp2, 2)
                            temp2{j} = temp2{j}(idx);
                        end
                    end
                    
                    %proc_events = [proc_events; temp1, temp2];
                    proc_events(outlinenum, 1:3) = temp1;
                    proc_events(outlinenum, 4:8) = temp2;
                    outlinenum = outlinenum + 1;
                    clear temp2;
                end
            else
                %fprintf(1, 'SKIP %s lines %s\n', proc_type, cell2mat(temp1{3}));
            end
            clear temp1;
            linenum= linenum+1;
            if (linenum < numlines)
                tline = r{linenum};
            else
                tline = '';
            end
        end
    elseif ischar(tline) && (strcmp(tline, 'v2') == 1)
        proc_events = cell(size(r, 1) - 1, 8);

        % processing version 2 (has annotation output, which is just the
        % size of the data being outputted.
        linenum = linenum + 1;
        if (linenum < numlines)
            tline = r{linenum};  
        else
            tline = '';
        end
        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            if isempty(find(strcmp(temp1{3}, proc_type), 1)) || ~isempty(find(strcmp('*', proc_type), 1))
                if (length(tline(pos+1:end)) > 0)
                    temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64 %u64', 'delimiter', ',', 'emptyvalue', 0);

                    %proc_events = [proc_events; temp1, temp2];
                    proc_events(outlinenum, 1:3) = temp1;
                    proc_events(outlinenum, 4:8) = temp2;
                     outlinenum = outlinenum + 1;
                   clear temp2;
                end
            else
                %fprintf(1, 'SKIP non worker lines %s\n', cell2mat(temp1{3}));
            end
            clear temp1;
            linenum = linenum + 1;
            if (linenum < numlines)
                tline = r{linenum};
            else
                tline = '';
            end
        end        
    else
        proc_events = cell(size(r, 1), 7);

        while ischar(tline) && ~isempty(strfind(tline, 'pid'))
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            if isempty(find(strcmp(temp1{3}, proc_type), 1)) || ~isempty(find(strcmp('*', proc_type), 1))
                if (length(tline(pos+1:end)) > 0)
                    temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64', 'delimiter', ',');

                    %proc_events = [proc_events; temp1, temp2];
                    proc_events(outlinenum, 1:3) = temp1;
                    proc_events(outlinenum, 4:7) = temp2;
                     outlinenum = outlinenum + 1;
                   clear temp2;
                end
            else
                %fprintf(1, 'SKIP non worker lines %s\n', cell2mat(temp1{3}));
            end
            clear temp1;
            linenum = linenum + 1;
            if (linenum < numlines)
                tline = r{linenum};
            else
                tline = '';
            end
        end
    end
    clear tline;
    clear r;
    toc
    
    
    %% merge action and comm on the same node
    
    tic;
    fprintf(1, 'merging cells\n');
    pids=[proc_events{:, 1}];   % pre-concatenate.  HUGE savings.
    nodes = unique(pids);
    pes2 = cell(size(nodes, 2), size(proc_events, 2));
    
    for j = 1:size(nodes, 2)
        pos = find(pids == nodes(j));
        pe2 = proc_events(pos, :);
        pes2{j, 1} = nodes(j);
        pes2{j, 2} = pe2{1, 2};
        pes2{j, 3} = pe2{1, 3};
        for i = 4:size(proc_events, 2)
            pes2{j, i} = cat(1, pe2{:, i});
        end
        
    end
    proc_events = pes2;
    toc

    function readPerf
        fn = 'TCGA.separate.jaguar.p10240.f100000.MPI_AMR.b4.io7680-16.is7680-1.csv'
        dir = '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1/async'

        % 0.61 sec on the input.
        tic;
        fid = fopen([dir '/' fn]);
        r = textscan(fid, '%s', 'Delimiter', '\n', 'bufsize', 21000000);
        fclose(fid);
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
           %# grow s if necessary
           if lineCt > sizS
               s = [s;cell(sizB,1)];
               sizS = sizS + sizB;
           end
           tline = fgetl(fid);
        end
        %# remove empty entries in s
        s(lineCt:end) = [];
        fclose(fid);
        toc
    end
    
    


end

 