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
    i = 1;
    while ischar(tline) & length(strfind(tline, 'pid')) > 0
        [temp1 pos] = textscan(tline, '%*s %d %*s %s %*s %s', 1, 'delimiter', ',');
        temp2 = textscan(tline(pos+1:end), '%s %d %u64 %u64', 'delimiter', ',');

        proc_events = [proc_events; temp1, temp2];
        clear temp1;
        clear temp2;
        tline = fgetl(fid);
        i = i+1;
    end
    clear tline;
    clear i;
    fclose(fid);
    
end

 