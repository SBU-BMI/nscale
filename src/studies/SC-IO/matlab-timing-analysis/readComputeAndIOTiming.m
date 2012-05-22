function [ header type startTimes endTimes location ] = readComputeAndIOTiming( filename )
%readComputeAndIOTiming reads the timing file
%   format of timing file is :
%       pid, hostname, filename, stagename, stagetype, stagename, stagetype, ...
    fid = fopen(filename);
    
    %first line is header
    % skip first 3.
    textscan(fid, '%*s %*s %*s', 1, 'delimiter', ','); 
    % then read the rest
    temp = textscan(fid, '%s %d', 'delimiter', ',');
    type = temp{1,2};
    numFields = length(type);
    header = temp{1,1}(1:numFields);
    fclose(fid);
    
    % next get the locations
%     fid = fopen(filename);
%     location = textscan(fid, '%d %s %*[^\n]', 'delimiter', ',', 'HeaderLines', 1); 
%     fclose(fid);
%     
    % now get the times
    startstr = '%d %s %*s ';
    timestr = repmat('%u64 %u64', 1, numFields);
    endstr = '%*[^\n]';
    formatstr = strcat(startstr, timestr, endstr);
    
    fid = fopen(filename);
    temp = textscan(fid, formatstr, 'delimiter', ',', 'HeaderLines', 1, 'CollectOutput', 1);
    location = temp(1, 1:2);
    startTimes = temp{1,3}(:, 1:2:end);
    endTimes = temp{1,3}(:, 2:2:end);
    
    fclose(fid);
end

 