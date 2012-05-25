function [ proc_events ] = readComputeAndIOTimingOneLineFormat( filename )
%readComputeAndIOTiming reads the timing file
%   format of timing file is :
%       pid, hostname, filename, stagename, stagetype, stagename, stagetype, ...
%       next line: values
%   then repeat.
    proc_events = {};

    fid = fopen(filename);
    
    % iterate through the lines
    tline = fgetl(fid);
    if ischar(tline) & length(strfind(tline, 'v2')) > 0        
        % processing version 2 (has annotation output, which is just the
        % size of the data being outputted.
        tline = fgetl(fid);  % skip header
        while ischar(tline) & length(strfind(tline, 'pid')) > 0
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64 %u64', 'delimiter', ',', 'emptyvalue', 0);

            proc_events = [proc_events; temp1, temp2];
            clear temp1;
            clear temp2;
            tline = fgetl(fid);
        end        
    else
        while ischar(tline) & length(strfind(tline, 'pid')) > 0
            [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
            temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64', 'delimiter', ',');

            proc_events = [proc_events; temp1, temp2];
            clear temp1;
            clear temp2;
            tline = fgetl(fid);
        end
    end
    clear tline;
    fclose(fid);
    
end

 