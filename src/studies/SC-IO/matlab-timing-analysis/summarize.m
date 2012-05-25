function [sampled_events, sum_events, data_sizes, sum_data ] = summarize( proc_events, sample_interval )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if (sample_interval > 1000000)
        printf(1, 'ERROR: sample interval should not be greater than 1000000 microseconds');
        return;
    end
    
    p = size(proc_events, 1);

    mn = inf;
    mx = 0;
    for i = 1:p 
	   mn = min([mn, min(proc_events{i, 6})]);
       mx = max([mx, max(proc_events{i, 7})]);
    end

    
    % init variables
    nTiles = zeros(p, 1);
    readTotals = zeros(p, 1);
    readDurations = [];
    memTotals = zeros(p, 1);
    memDurations = [];
    computeTotals = zeros(p, 1);
    computeDurations = [];
    barrierTotals = zeros(p, 1);
    barrierDurations = [];

    writeTotals = zeros(p, 1);
    writeDurations = [];
    awriteTotals = zeros(p, 1);
    awriteDurations = [];
    acloseTotals = zeros(p, 1);
    acloseDurations = [];
    
    % per node info
    for i = 1:p-1
        names = proc_events{i, 4};
        types = proc_events{i, 5};
        startt = proc_events{i, 6};
        endt = proc_events{i, 7};

        % tile total
        nTiles(i) = sum(types == 0, 1);
        
        % compute total
        idx = find(types == 0);
        duration = endt(idx) - startt(idx);
        computeTotals(i) = sum(duration,1);
        computeDurations = [computeDurations; duration];
        % mem total
        idx = find(types == 11);
        duration = endt(idx) - startt(idx);
        memTotals(i) = sum(duration,1);
        memDurations = [memDurations; duration];

        
        % read total
        idx = find(types == 31);
        duration = endt(idx) - startt(idx);
        readTotals(i) = sum(duration,1);
        readDurations = [readDurations; duration];

        
        % barrier total
        idx = find(strcmp(names, 'IO MPI scan') | strcmp(names, 'IO MPI allreduce'));
        duration = endt(idx) - startt(idx);
        barrierTotals(i) = sum(duration,1);
        barrierDurations = [barrierDurations; duration];
        
    
        % adios write total
        idx = find( types == 44 );
        duration = endt(idx) - startt(idx);
        awriteTotals(i) = sum(duration,1);
        awriteDurations = [awriteDurations; duration];
        
        % adios write total
        idx = find( types == 45 );
        duration = endt(idx) - startt(idx);
        acloseTotals(i) = sum(duration,1);
        acloseDurations = [acloseDurations; duration];
        
        % write total
        idx = find( types == 32 );
        duration = endt(idx) - startt(idx);
        writeTotals(i) = sum(duration,1);
        writeDurations = [writeDurations; duration];

    
    end
    
    allEventTypes = [-1 0 11 12 21 22 31 32 41 42 43 44 45 46]';
    [sampled_events, sum_events, data_sizes, sum_data] = sampleEvents(proc_events, p, mx, sample_interval, allEventTypes);
    % find the 1 sec interval during whick we have max IO.
    
    readTotal = sum(readTotals, 1);
    readPercent = readTotal / double(mx * p);
    readMean = mean(readDurations);
    readStdev = std(double(readDurations));
    
    memTotal = sum(memTotals, 1);
    memPercent = memTotal / double(mx * p);
    memMean = mean(memDurations);
    memStdev = std(double(memDurations));
    
    computeTotal = sum(computeTotals, 1);
    computePercent = computeTotal / double(mx * p);
    computeMean = mean(computeDurations);
    computeStdev = std(double(computeDurations));
    
    barrierTotal = sum(barrierTotals, 1);
    barrierPercent = barrierTotal / double(mx * p);
    barrierMean = mean(barrierDurations);
    barrierStdev = std(double(barrierDurations));
    
    
    writeTotal = sum(writeTotals, 1);
    writePercent = writeTotal / double(mx * p);
    writeMean = mean(writeDurations);
    writeStdev = std(double(writeDurations));
    
    awriteTotal = sum(awriteTotals, 1);
    awritePercent = awriteTotal / double(mx * p);
    awriteMean = mean(awriteDurations);
    awriteStdev = std(double(awriteDurations));
    
    acloseTotal = sum(acloseTotals, 1);
    aclosePercent = acloseTotal / double(mx * p);
    acloseMean = mean(acloseDurations);
    acloseStdev = std(double(acloseDurations));
    
    % average within a 1 second sliding window to get the throughput
    window = ones(round(1000000 / sample_interval), 1, 'double');
    
    % throughputs in gigabytes
    scaling = 1 / double(1024*1024*1024); % GB/s
    memMaxThroughPut = max(conv(full(sum_data(:,3)), window, 'same')) * scaling;
    readMaxThroughPut = max(conv(full(sum_data(:,7)), window, 'same')) * scaling;
    iwriteMaxThroughPut = max(conv(full(sum_data(:,8)), window, 'same')) * scaling;
    adiosWriteMaxThroughPut = max(conv(full(sum_data(:,12)), window, 'same')) * scaling;
    adiosCloseMaxThroughPut = max(conv(full(sum_data(:,13)), window, 'same')) * scaling;

    fprintf(1, 'Operation,TotalTime(ms),PercentTime,MeanTime(ms),StdevTime(ms),Throughput(GB/s)\n');
    fprintf(1, 'Read,%f,%f,%f,%f,%f\n', readTotal/1000, readPercent, readMean/1000, readStdev/1000, readMaxThroughPut);
    fprintf(1, 'Compute,%f,%f,%f,%f,\n', computeTotal/1000, computePercent, computeMean/1000, computeStdev/1000);
    fprintf(1, 'Barrier,%f,%f,%f,%f,\n', barrierTotal/1000, barrierPercent, barrierMean/1000, barrierStdev/1000);
    fprintf(1, 'Image Write,%f,%f,%f,%f,%f\n', writeTotal/1000, writePercent, writeMean/1000, writeStdev/1000, iwriteMaxThroughPut);
    fprintf(1, 'Adios Write,%f,%f,%f,%f,%f\n', awriteTotal/1000, awritePercent, awriteMean/1000, awriteStdev/1000, adiosWriteMaxThroughPut);
    fprintf(1, 'Adios close,%f,%f,%f,%f,%f\n', acloseTotal/1000, aclosePercent, acloseMean/1000, acloseStdev/1000, adiosCloseMaxThroughPut);
    fprintf(1, 'Mem,%f,%f,%f,%f,%f\n', memTotal/1000, memPercent, memMean/1000, memStdev/1000, memMaxThroughPut);
    
    
    % compute the write throughput : 
    % need to know how many tiles were processed.
end




function [ sampled_events sum_events data_sizes sum_data ] = sampleEvents (proc_events, p, mx, sample_interval, allEventTypes)
    % p is number of cells (procs)
    % mx is maximum timestamp overall;

    % get the number of pixels
    cols = ceil(double(mx) / double(sample_interval));
    
    % allocate norm-events
	sampled_events = cell(p, 1);
    data_sizes = cell(p, 1);
    
    % generate the sampled_events
    for i = 1:p
        sampled_events{i} = sparse(cols, length(allEventTypes));
        data_sizes{i} = sparse(cols, length(allEventTypes));
        
        types = proc_events{i, 5};
        [blah typeIdx] = ismember(types, allEventTypes);
        clear blah;
        startt = double(proc_events{i, 6});
        endt = double(proc_events{i, 7});
        datasize = proc_events{i, 8};
        
        startt_bucket = ceil(double(startt) / double(sample_interval));
        endt_bucket = ceil(double(endt) / double(sample_interval));
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;
%        start_bucket_start = (startt_bucket - 1) * sample_interval + 1;
%        end_bucket_end = endt_bucket * sample_interval;

        duration = double(endt - startt + 1);  % data rate per microsec.
        

        % can't do this in more vectorized way, because of the range access
        for j = 1:length(types)
%           not faster...
%             % first mark the entire range
%             sampled_events{i}(startt_bucket(j) : endt_bucket(j), typeIdx(j)) = ...
%                 sampled_events{i}(startt_bucket(j) : endt_bucket(j), typeIdx(j)) + sample_interval;
%             % then remove the leading
%             sampled_events{i}(startt_bucket(j), typeIdx(j)) = ...
%                 sampled_events{i}(startt_bucket(j), typeIdx(j)) - (startt(j) - start_bucket_start(j) + 1);
%             % then remove the trailing
%             sampled_events{i}(endt_bucket(j), typeIdx(j)) = ...
%                 sampled_events{i}(endt_bucket(j), typeIdx(j)) - (end_bucket_end(j) - endt(j) + 1);
    

           % if start and end in the same bucket, mark with duration
           if startt_bucket(j) == endt_bucket(j)
               sampled_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   sampled_events{i}(startt_bucket(j), typeIdx(j)) + endt(j) - startt(j) + 1;
               data_sizes{i}(startt_bucket(j), typeIdx(j)) = ...
                   data_sizes{i}(startt_bucket(j), typeIdx(j)) + double(datasize(j));
           else
               % do the start first
               t = start_bucket_end(j) - startt(j) + 1;
               sampled_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   sampled_events{i}(startt_bucket(j), typeIdx(j)) + t;
               data_sizes{i}(startt_bucket(j), typeIdx(j)) = ...
                   data_sizes{i}(startt_bucket(j), typeIdx(j)) + double(datasize(j)) * double(t) / duration(j);
                % then do the end
               t = endt(j) - end_bucket_start(j) + 1;
               sampled_events{i}(endt_bucket(j), typeIdx(j)) = ...
                    sampled_events{i}(endt_bucket(j), typeIdx(j)) + t;
               data_sizes{i}(endt_bucket(j), typeIdx(j)) = ...
                   data_sizes{i}(endt_bucket(j), typeIdx(j)) + double(datasize(j)) * double(t) / duration(j);

               % then do in between
               if endt_bucket(j) > (startt_bucket(j) + 1)
                    sampled_events{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) = ...
                        sampled_events{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) + sample_interval;
                   data_sizes{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) = ...
                       data_sizes{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) + double(datasize(j)) * double(sample_interval) / duration(j);
               end
           end 
         end
    	sampled_events{i} = sampled_events{i} / double(sample_interval);
        
        clear types;
        clear startt;
        clear endt;
        clear startt_bucket;
        clear endt_bucket;
        clear start_bucket_end;
        clear end_bucket_start;
        
    end
    
    sum_events = sparse(cols, length(allEventTypes));
    sum_data = sparse(cols, length(allEventTypes));
	for i = 1:p	
		sum_events = sum_events + sampled_events{i};
		sum_data = sum_data + data_sizes{i};
    end

end
