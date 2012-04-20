function [ concurrencyAll nodes concurrencyPerNode ] = ComputeAndIOConcurrency( header, type, startTimes, endTimes, location )
%ComputeAndIOConcurrency Compute the level of concurrency
%   header just gives information about start and end (1xN)
%   type specifies whether it's computations (0), IO (2), admin/idle (-1). (1xN)
%   startTimes and endTimes are the timestamps (MxN)
%   location is the processId and hostname.  (Mx2 cell array)

% per node
    nodes = unique(location{:, 2});
    concurrencyPerNode = cell(length(nodes), 1);
    for i = 1 : length(nodes)
       %get the rows corresponding to the host
       idx = find(strcmp(location{:, 2}, nodes(i, 1)) == 1);
       nodeProcs = length(unique(location{:, 1}(idx)));  % count ranks for host
       
       concurrencyPerNode{i, 1} = computeConcurrency(header, type, startTimes(idx, :), endTimes(idx, :), nodeProcs, 0);
    end

% all processes
    nodeProcs = length(unique(location{:, 1}));  % count the unique ranks.
    concurrencyAll  = computeConcurrency(header, type, startTimes, endTimes, nodeProcs, 1);

end

function [ output ] = computeConcurrency(header, type, stime, etime, cpuCount, computeAverage)
    alltimes = reshape([stime; etime], 1, size(stime, 1) * size(stime, 2) + size(etime, 1) * size(etime, 2));
    alltimes = unique(alltimes);
    numEvents = length(alltimes);
    
    timestamps = sort(alltimes);
    concurrency = zeros(3, numEvents, 'int64');
    
    compute = (type' == 0);
    io = (type' == 2);
    for i = 1 : size(stime, 1)
       % compute
       active = int64(ismember(timestamps, stime(i, compute )));
       concurrency(2, :) = concurrency(2, :) + active;
       concurrency(1, :) = concurrency(1, :) - active;
       % compute
       active = int64(ismember(timestamps, etime(i, compute )));
       concurrency(2, :) = concurrency(2, :) - active;
       concurrency(1, :) = concurrency(1, :) + active;
         
       % IO
       active = int64(ismember(timestamps, stime(i, io )));
       concurrency(3, :) = concurrency(3, :) + active;
       concurrency(1, :) = concurrency(1, :) - active;
       % IO
       active = int64(ismember(timestamps, etime(i, io )));
       concurrency(3, :) = concurrency(3, :) - active;
       concurrency(1, :) = concurrency(1, :) + active;        
         
    end
    
%     for i = 1: length(timestamps);
%        % for each point where there is an event.
%        % find what the event type is.  count the total number, and
%        % increment for start, decrement for end
%        
%        % find colmns (events) starting at this timestamp
%        a = sum(stime == timestamps(i), 1);  % number of active in each stage
%        active = sum(a .* (type' == 0)); % number of active in compute
%        concurrency(2, i) = concurrency(2, i) + active;
%        concurrency(1, i) = concurrency(1, i) - active;
%        
%        active = sum(a .* (type' == 2)); % number of active in IO
%        concurrency(3, i) = concurrency(3, i) + active;
%        concurrency(1, i) = concurrency(1, i) - active;
%        
%        
%        % find colmns (events) ending at this timestamp
%        a = sum(etime == timestamps(i), 1);  % number of active exiting each stage
%        active = sum(a .* (type' == 0)); % number of active in compute
%        concurrency(2, i) = concurrency(2, i) - active;
%        concurrency(1, i) = concurrency(1, i) + active;
%        
%        active = sum(a .* (type' == 2)); % number of active in IO
%        concurrency(3, i) = concurrency(3, i) - active;
%        concurrency(1, i) = concurrency(1, i) + active;
%        
%        
%     end

    % prepend time 0
    timestamps = [0 timestamps];
    concurrency = [[cpuCount; 0; 0] concurrency];
    
    % now do scan to compute the actual concurrency
    for i = 2: length(timestamps)
       concurrency(:, i) = concurrency(:, i-1) + concurrency(:, i); 
    end
    
    if (computeAverage == 1)
        duration = double(timestamps(1,end) - timestamps(1,1));
        intervals = double(timestamps(1,2:end) - timestamps(1, 1:end-1));
        totaltime = dot(double(concurrency(3,1:end-1)), intervals);
        averageIO = totaltime / duration
        nodetimeaverageIO = averageIO / double(cpuCount)
        totaltime = dot(double(concurrency(2,1:end-1)), intervals);
        averageCompute = totaltime / duration
        nodetimeaverageCompute = averageCompute / double(cpuCount)
    end
    
    % turn into intervals
    t2 = zeros(1, size(timestamps, 2) * 2 - 1);
    t2(1, 1:2:end) = timestamps(1, :);
    t2(1, 2:2:end) = timestamps(1, 2:end);
    c2 = zeros(size(concurrency, 1), size(concurrency, 2) * 2 - 1);
    c2(:, 1:2:end) = concurrency(:, :);
    c2(:, 2:2:end) = concurrency(:, 1:end-1);
    
    output = cell(1,2);
    output{1, 1} = t2;
    output{1, 2} = c2;
end