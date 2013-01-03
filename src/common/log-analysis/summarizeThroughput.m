function [throughputs, ops] = summarizeThroughput( dataPerInterval, sample_interval)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %% check the parameters.

    ops = {'PeakTP(GB/s)', 'MeanTP(GB/s)', 'StdevTP(GB/s)'};

    
    %% first get some "constants"
    % MATLAB:  unique on dataset and maybe dataset concat is slow.  operate
    % on originam event_names first.
    p = size(dataPerInterval, 1);
    % get the unique event names, type mapping, and also the count

    num_ev_vals = size(dataPerInterval{1}, 2);
     
    scaling = 1.0 / double(1024*1024*1024); % GB/s
    kernel = ones(round(1000000.0 / sample_interval), 1, 'double');
    cols = size(dataPerInterval{1}, 1);
    
    %% now compute event stats by name

    ndata_time = sparse(cols, num_ev_vals);

    for i = 1:p
        ndata_time = ndata_time + dataPerInterval{i};
    end

    % compute throughput
    TPIn1sec = zeros(cols, num_ev_vals);
    %avgTPAvgNode = zeros(1, unique_count);
    
    % find the 1 sec interval during whick we have max IO.
    for n = 1:num_ev_vals
        TPIn1sec(:, n) = conv(full(ndata_time(:,n)), kernel, 'same') * scaling;
    end

    throughputs = zeros(num_ev_vals, length(ops));
    
    throughputs(:, 1) = max(TPIn1sec, [], 1)';
    throughputs(:, 2) = mean(TPIn1sec, 1)';
    throughputs(:, 3) = std(TPIn1sec, 0, 1)';

    
    clear ndata_time;
    clear num_ev_vals;
    clear TPIn1sec;
    clear p;
    clear scaling;
    clear cols;
    clear kernel;
end


