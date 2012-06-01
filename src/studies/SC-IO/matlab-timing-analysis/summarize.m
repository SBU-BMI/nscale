function [] = summarize( proc_events, sample_interval, fid, proc_type, allEventTypes, allTypeNames)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %% check the parameters.

    if (isempty(proc_events))
        printf(2, 'ERROR: no events to process\n');
        return;
    end
    
    if (isempty(fid) || fid < 1)
        fid = 1;
    end
    
    if (isempty(proc_type))
        proc_type = 'w';
    end
    
    % get some information about the events
    % filter "in" the workers only.
    % also calculate the max timestamp.
    pidx = [];
    mx = [];
    for i = 1:size(proc_events, 1) 
        if strcmp(proc_type, proc_events{i,3})
            pidx = [pidx; i];
        end
        mx = [mx, max(proc_events{i, 7})];  % maximum end timestamp
    end
    mx = max(mx);
    events = proc_events(pidx, :);
    
    % now check the sampling interval
    if (isempty(sample_interval))
       sample_interval = min(mx / 50000, 1000000);
    end
    if (sample_interval > 1000000)
        printf(2, 'ERROR: sample interval should not be greater than 1000000 microseconds\n');
        return;
    end

    
    %% first get some "constants"
    % MATLAB:  unique on dataset and maybe dataset concat is slow.  operate
    % on originam event_names first.
    p = size(events, 1);
    % get the unique event names, type mapping, and also the count
    event_names = {};
    event_types = [];
    for i = 1:p
        event_names = [event_names; events{i,4}];
        event_types = [event_types; events{i,5}];
        [event_names ia ic] = unique(event_names);
        event_types = event_types(ia);
    end
    clear ia;
    clear ic;
    unique_types = unique(event_types);
    num_ev_names = size(event_names, 1);
    num_ev_types = size(unique_types, 1);
     
    scaling = 1 / double(1024*1024*1024); % GB/s
    hasDataSizes = size(events, 2) > 7;
    kernel = ones(round(1000000 / sample_interval), 1, 'double');
    cols = ceil(double(mx) / double(sample_interval));
    
    % application specific computation.
    num_tiles = zeros(p, 1);
    for i = 1:p
       num_tiles(i) = sum(events{i, 5} == 0, 1); 
    end
    
    
    %% now compute event stats by name
    % get the total durations of each event by name
    % using find(strcmp()) in an inner loop for the names is faster - 0.95s
    % vs 1.7 sec if using unique and intersect to get the mappings.
    ntotals = zeros(p, num_ev_names);
    ndurations = cell(num_ev_names, 1);
    for i = 1:p
        names = events{i, 4};
        duration = events{i, 7} - events{i, 6} + 1;
        for n = 1:num_ev_names
            % compute total
            idx = find(strcmp(event_names(n), names));
            dur = duration(idx);
            ntotals(i, n) = sum(dur,1);
            ndurations{n} = [ndurations{n}; dur];
        end
    end
    clear duration;
    clear names;
    clear idx;
    clear dur;
    
    if (hasDataSizes)
        [ndata_time ndata_proc ndata_sizes] = resampleDataByName(events, p, cols, sample_interval, event_names);
    else
        ndata_time = sparse(cols, num_ev_names);
        ndata_proc = zeros(p, num_ev_names);
    end
        
    computeTimeStats(fid, ntotals, ndurations, mx, p, event_names, hasDataSizes, ndata_time, kernel, scaling, ndata_proc);
    
    
    %% then analyze by event type
    
    
    % ge tthe total durations of each event by type
    ttotals = zeros(p, num_ev_types);
    tdurations = cell(num_ev_types, 1);
    for n = 1:num_ev_types
        % compute total
        idx = find(event_types == unique_types(n));
        ttotals(:, n) = sum(ntotals(:, idx), 2);
        tdurations{n} = cat(1, ndurations{idx});
    end
    
    
    % aggregate to ge tdata sizes
    tdata_time = sparse(cols, num_ev_types);
    tdata_proc = zeros(p, num_ev_types);
    if (hasDataSizes)
        for i = 1:p
            tdata_sizes = sparse(cols, num_ev_types);
            for n = 1:num_ev_types
                idx = find(event_types == unique_types(n));
                tdata_sizes(:, n) = sum(ndata_sizes{i}(:, idx), 2);
                tdata_proc(i, n) = sum(ndata_proc(i, idx), 2);
            end
            tdata_time = tdata_time + tdata_sizes;
        end
        clear id;
        clear tdata_sizes;
    end
    
    [lia locb] = ismember(unique_types, allEventTypes);
    typenames = allTypeNames(locb);
    clear lia;
    clear locb;
    computeTimeStats(fid, ttotals, tdurations, mx, p, typenames, hasDataSizes, tdata_time, kernel, scaling, tdata_proc);

    
    
    %% finally analyze by user specified grouping of events
    
    % group by custom name list
    % first column is the time events to include. second column is the data
    % event to include.
    time_data_names = { {'IO define vars', ...
        'IO malloc vars', 'IO tile metadata', 'IO tile data', ...
        'IO MPI scan', 'IO MPI allreduce', 'IO var clear'}, ...
        {'IO tile data'}; ...
        {'adios open', 'ADIOS WRITE', 'ADIOS WRITE Summary', ...
            'adios close'}, {'adios close'}};
    
    labels = {'AdiosPrep', 'Adios'};    
    
    
    time_data_idx = cell(size(time_data_names));
    for i = 1:size(time_data_names, 1)
        for j = 1:size(time_data_names,2)
            [lia locb] = ismember(time_data_names{i, j}, event_names);
            time_data_idx{i, j} = locb;
        end
    end

    num_comp_events = length(labels);
    
    ctotals = zeros(p, num_comp_events);
    cdurations = cell(num_comp_events, 1);
    for n = 1:num_comp_events
        
        if (~isempty(time_data_idx{n, 1}) & ~isempty(time_data_idx{n, 2}))
            ctotals(:, n) = sum(ntotals(:, time_data_idx{n, 1}), 2);
            cdurations{n} = cat(1, ndurations{time_data_idx{n,1}});
        end
    end
    
    % aggregate to ge tdata sizes
    cdata_time = sparse(cols, num_comp_events);
    cdata_proc = zeros(p, num_comp_events);
    if (hasDataSizes)
        cdata_sizes = cell(p, 1);
        for i = 1:p
            cdata_sizes{i} = sparse(cols, num_comp_events);
            for n = 1:num_comp_events
                cdata_sizes{i}(:, n) = sum(ndata_sizes{i}(:, time_data_idx{n,2}), 2);
                cdata_proc(i, n) = sum(ndata_proc(i, time_data_idx{n,2}), 2);
            end
            cdata_time = cdata_time + cdata_sizes{i};
        end
        clear cdata_sizes;
    end

    computeTimeStats(fid, ctotals, cdurations, mx, p, labels, hasDataSizes, cdata_time, kernel, scaling, cdata_proc);

end

function computeTimeStats(fid, totals, durations, tmax, p, labels, hasDataSizes, data_time, kernel, scaling, data_proc) 
    label_count = length(labels);

    tot = sum(totals, 1);
    mx = max(totals, [], 1);
    mn = min(totals, [], 1);
    av = mean(totals, 1);
    med = median(totals, 1);
    mo = mode(totals, 1);

    percent = tot / double(tmax * p);
    eventmean = zeros(label_count,1);
    eventstdev = zeros(label_count,1);
    eventmedian = zeros(label_count,1);
    eventmode = zeros(label_count,1);
    eventmin = zeros(label_count,1);
    eventmax = zeros(label_count,1);
    for n = 1:label_count
        eventmean(n) = mean(durations{n});
        eventstdev(n) = std(double(durations{n}));
        eventmedian(n) = median(double(durations{n}));
        eventmode(n) = mode(double(durations{n}));
        eventmin(n) = min(durations{n});
        eventmax(n) = max(durations{n});
    end

    if hasDataSizes
    
        % compute throughput
        maxTPIn1sec = zeros(label_count, 1);
        %avgTPAvgNode = zeros(1, unique_count);

        % find the 1 sec interval during whick we have max IO.
        for n = 1:label_count
            maxTPIn1sec(n) = max(conv(full(data_time(:,n)), kernel, 'same')) * scaling;
        end
        data_t = sum(data_proc, 1) * 1000000 * scaling;
        % avg TP per node is total data / total CPU time.
        %navgTPAvgNode = ndata_t ./ ntotal * 1000000 * scaling ;
        % avg TP is total data / average CPU time
        avgTP = data_t ./ av;
        minTP = data_t ./ mx;
        maxTP = data_t ./ mn;
        medianTP = data_t ./ med;
        modeTP = data_t ./ mo;
        
        tp = double(data_proc) ./ double(totals) * 1000000 * scaling;
        TPtotal = sum(tp, 1);
        TPmean = mean(tp, 1);
        TPmedian = median(tp, 1);
        TPmode = mode(tp,1);
        TPstdev = std(tp, 1);
        TPmin = min(tp, [], 1);
        TPmax = max(tp, [], 1);
        clear tp;
        
    end
    
    
    fprintf(fid, 'Operation,TotalTime(ms),PercentTime,MeanTime,StdevTime,MedianTime,ModeTime,MinTime,MaxTime');
    if hasDataSizes
        fprintf(fid, ',MaxTPin1sec(GB/s),avgTP=sum(data)/mean(t),medianTP,modeTP,minTP,maxTP,nodeTPtotal=sum(data/t),nodeTPavg,nodeTPstdev,nodeTPmedian,nodeTPmode,nodeTPmin,nodeTPmax');
    end
    fprintf(fid, '\n');
    
    if hasDataSizes
        for n = 1:label_count
            fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                labels{n}, tot(n)/1000, percent(n), ...
                eventmean(n)/1000, eventstdev(n)/1000, ...
                eventmedian(n)/1000, eventmode(n)/1000,...
                eventmin(n)/1000, eventmax(n)/1000,...
                maxTPIn1sec(n), ...
                avgTP(n), medianTP(n), modeTP(n),...
                minTP(n), maxTP(n), ...
                TPtotal(n),TPmean(n),TPstdev(n),TPmedian(n),TPmode(n),...
                TPmin(n),TPmax(n));
        end
    else 
        for n = 1:label_count
            fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                labels{n}, tot(n)/1000, percent(n), ...
                eventmean(n)/1000, eventstdev(n)/1000, ...
                eventmedian(n)/1000, eventmode(n)/1000,...
                eventmin(n)/1000, eventmax(n)/1000);
        end
    end
    fprintf(fid, '\n');

end


function [ ndata_time ndata_proc ndata_sizes] = ...
    resampleDataByName(events, p, cols, sample_interval, event_names)
    % p is number of cells (procs)
    % mx is maximum timestamp overall;

    % get the number of pixels
    num_ev_names = length(event_names);
    
    % allocate norm-events
	ndata_sizes = cell(p, 1);
    ndata_proc = zeros(p, num_ev_names);
    
    % generate the sampled_events
    for i = 1:p

        ndata_sizes{i} = sparse(cols, num_ev_names);
        
        names = events{i, 4};
        startt = double(events{i, 6});
        endt = double(events{i, 7});
        datasize = double(events{i, 8});
        
        [blah idx] = ismember(names, event_names);
        clear blah;
        
        startt_bucket = ceil(double(startt) / double(sample_interval));
        endt_bucket = ceil(double(endt) / double(sample_interval));
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;
        start_bucket_start = (startt_bucket - 1) * sample_interval + 1;
        end_bucket_end = endt_bucket * sample_interval;

        duration = double(endt - startt + 1);  % data rate per microsec.
        
        for j = 1:num_ev_names
            ndata_proc(i, j) = sum(datasize(find(idx == j)),1);
        end

        datarate = datasize ./ duration;
        startdr = datarate .* double(start_bucket_end - startt + 1);
        enddr = datarate .* double(endt - end_bucket_start + 1);
        fulldr = datarate * double(sample_interval);
        % can't do this in more vectorized way, because of the range access
        tmpdata = zeros(cols, num_ev_names);
        for j = 1:length(names)
            x1 = startt_bucket(j);
            x2 = endt_bucket(j);
            y = idx(j);
            
           % if start and end in the same bucket, mark with duration
           if x1 == x2
               tmpdata(x1, y) = ...
                   tmpdata(x1, y) + datasize(j);
           else
               % do the start first
               tmpdata(x1, y) = ...
                   tmpdata(x1, y) + startdr(j);
                % then do the end
               tmpdata(x2,y) = ...
                   tmpdata(x2, y) + enddr(j);

               % then do in between
               if x2 > (x1 + 1)
                    tmpdata(x1+1 : x2-1, y) = ...
                       tmpdata(x1+1 : x2-1, y) + fulldr(j);
               end
           end 
        end
        
        startdr2 = datarate .* double(startt - start_bucket_start + 1);
        enddr2 = datarate .* double(end_bucket_end - endt + 1);
        tmpdata = zeros(cols, num_ev_names);
        for j = 1:length(names)
            x1 = startt_bucket(j);
            x2 = endt_bucket(j);
            y = idx(j);
            
           % if start and end in the same bucket, mark with duration
           if x1 == x2
               tmpdata(x1, y) = ...
                   tmpdata(x1, y) + datasize(j);
           else
               % then do in between
                tmpdata(x1 : x2, y) = ...
                   tmpdata(x1 : x2, y) + fulldr(j);

               % do the start first
               tmpdata(x1, y) = ...
                   tmpdata(x1, y) - startdr2(j);
                % then do the end
               tmpdata(x2,y) = ...
                   tmpdata(x2, y) - enddr2(j);
           end 
        end

        ndata_sizes{i} = sparse(tmpdata);
    	clear idx;
        clear names;
        clear startt;
        clear endt;
        clear startt_bucket;
        clear endt_bucket;
        clear start_bucket_end;
        clear end_bucket_start;
        clear datasize;
        clear duration;
    end

    
    ndata_time = sparse(cols, num_ev_names);
    for i = 1:p	
		ndata_time = ndata_time + ndata_sizes{i};
	end
    clear num_ev_names;
end
