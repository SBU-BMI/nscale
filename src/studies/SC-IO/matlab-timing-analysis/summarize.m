function summarize( proc_events, sample_interval, fid, proc_type, allEventTypes, allTypeNames)
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
        proc_type = '*';
    end
    
    % get some information about the events
    % filter "in" the workers only.
    % also calculate the max timestamp.
    mx = 0;
    for i = 1:size(proc_events, 1) 
        mx = max([mx, max(proc_events{i, 7})]);  % maximum end timestamp
    end
    if strcmp(proc_type, '*')
        events = proc_events;
    else 
        pidx = strcmp(proc_type, cat(1, proc_events{:, 3}));
        events = proc_events(pidx, :);
        clear pidx;
    end
    
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
        [event_names, ia, ~] = unique(event_names);
        event_types = event_types(ia);
        clear ia;
    end
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
            idx = strcmp(event_names(n), names);
            dur = duration(idx);
            ntotals(i, n) = sum(dur,1);
            ndurations{n} = [ndurations{n}; dur];
            clear idx;
            clear dur;
        end
        clear names;
        clear duration;
    end
    
    ndata_time = sparse(cols, num_ev_names);
    ndata_proc = zeros(p, num_ev_names);
    ndata_sizes = cell(p, 1);

    % inline function shares variable so don't need to copy variables.
    function resampleDataByName2
        % p is number of cells (procs)
        % mx is maximum timestamp overall;


        % generate the sampled_events
        for i = 1:p

            ndata_sizes{i} = sparse(cols, num_ev_names);

            names = events{i, 4};
            startt = double(events{i, 6});
            endt = double(events{i, 7});
            datasize = double(events{i, 8});

            [~, idx] = ismember(names, event_names);
            clear names;

            startt_bucket = ceil(startt / sample_interval);
            endt_bucket = ceil(endt / sample_interval);

            start_bucket_end = startt_bucket * sample_interval;
            end_bucket_start = (endt_bucket - 1) * sample_interval + 1;

            duration = endt - startt + 1;  % data rate per microsec.

            for j = 1:num_ev_names
                ndata_proc(i, j) = sum(datasize(idx == j),1);
            end

            %datarate = datasize ./ duration;
            fulldr = (datasize * sample_interval) ./ duration;
            % can't do this in more vectorized way, because of the range access

            tmpdata = zeros(cols, num_ev_names);
            startdr = datasize .* (start_bucket_end - startt + 1) ./ duration;
            enddr = datasize .* (endt - end_bucket_start + 1) ./ duration;

            clear startt;
            clear endt;

            clear start_bucket_end;
            clear end_bucket_start;

            clear duration;

            for j = 1:length(datasize)
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
            clear x1;
            clear x2;
            clear y;
            clear startdr;
            clear enddr;
            clear fulldr;
            clear startt_bucket;
            clear endt_bucket;
            clear idx;
            clear datasize;


            ndata_sizes{i} = sparse(tmpdata);
            clear tmpdata;
        end

        for i = 1:p	
            ndata_time = ndata_time + ndata_sizes{i};
        end
        clear num_ev_names;
    end    
    
    if (hasDataSizes)
%        [ndata_time ndata_proc ndata_sizes] = resampleDataByName(events, p, cols, sample_interval, event_names);
        resampleDataByName2;
    end
        
    computeTimeStats(fid, ntotals, ndurations, mx, p, event_names, hasDataSizes, ndata_time, kernel, scaling, ndata_proc);
    clear ndata_time;
    clear num_ev_names;
    clear events;
    
    %% then analyze by event type
    
    
    % ge tthe total durations of each event by type
    ttotals = zeros(p, num_ev_types);
    tdurations = cell(num_ev_types, 1);
    for n = 1:num_ev_types
        % compute total
        idx = find(event_types == unique_types(n));
        ttotals(:, n) = sum(ntotals(:, idx), 2);
        tdurations{n} = cat(1, ndurations{idx});
        clear idx;
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
                clear idx;
            end
            tdata_time = tdata_time + tdata_sizes;
            clear tdata_sizes;
        end
    end
    
   
    [~, locb] = ismember(unique_types, allEventTypes);
    typenames = allTypeNames(locb);
    clear locb;
    computeTimeStats(fid, ttotals, tdurations, mx, p, typenames, hasDataSizes, tdata_time, kernel, scaling, tdata_proc);

    clear tdata_time;
    clear tdata_proc;
    clear ttotals;
    clear tdurations;
    clear typenames;
    clear num_ev_types;
    
    %% finally analyze by user specified grouping of events
    
    % group by custom name list
    % first column is the time events to include. second column is the data
    % event to include.
    time_data_names = { ...
        {'IO define vars', ...
        'IO malloc vars',...
        'IO tile metadata',...
        'IO tile data', ...
        'IO MPI scan', ...
        'IO MPI allreduce', ...
        'IO var clear'}, ...
        {'IO tile data'}; ...
        {'adios open', ...
        'ADIOS WRITE', ...
        'ADIOS WRITE Summary', ...
        'adios close'},...
        {'adios close'}; ...
        {'BENCH 0 IO define vars', ...
        'BENCH 0 IO malloc vars',...
        'BENCH 0 IO tile metadata',...
        'BENCH 0 IO tile data', ...
        'BENCH 0 IO MPI scan', ...
        'BENCH 0 IO MPI allreduce', ...
        'BENCH 0 IO var clear'}, ...
        {'BENCH 0 IO tile data'}; ...
        {'BENCH 0 adios open', ...
        'BENCH 0 ADIOS WRITE', ...
        'BENCH 0 ADIOS WRITE Summary', ...
        'BENCH 0 adios close'},...
        {'BENCH 0 adios close'}; ...
        {'BENCH 1 IO define vars', ...
        'BENCH 1 IO malloc vars',...
        'BENCH 1 IO tile metadata',...
        'BENCH 1 IO tile data', ...
        'BENCH 1 IO MPI scan', ...
        'BENCH 1 IO MPI allreduce', ...
        'BENCH 1 IO var clear'}, ...
        {'BENCH 1 IO tile data'}; ...
        {'BENCH 1 adios open', ...
        'BENCH 1 ADIOS WRITE', ...
        'BENCH 1 ADIOS WRITE Summary', ...
        'BENCH 1 adios close'},...
        {'BENCH 1 adios close'} ...
        };
    
    labels = {'AdiosPrep', 'Adios', 'Benchmark 0 Prep', 'Benchmark 0 ADIOS', 'Benchmark 1 Prep', 'Benchmark 1 ADIOS'};    
    
    
    time_data_idx = cell(size(time_data_names));
    for i = 1:size(time_data_names, 1)
        for j = 1:size(time_data_names,2)
            [~, locb] = ismember(time_data_names{i, j}, event_names);
            time_data_idx{i, j} = locb(locb > 0);
        end
    end
    clear time_data_names;
    
    num_comp_events = length(labels);
    
    ctotals = zeros(p, num_comp_events);
    cdurations = cell(num_comp_events, 1);
    for n = 1:num_comp_events
        
        if ~isempty(time_data_idx{n, 1}) 
            ctotals(:, n) = sum(ntotals(:, time_data_idx{n, 1}), 2);
            cdurations{n} = cat(1, ndurations{time_data_idx{n,1}});
        end
    end
    
    % aggregate to ge tdata sizes
    cdata_time = sparse(cols, num_comp_events);
    cdata_proc = zeros(p, num_comp_events);
    if (hasDataSizes)
        for i = 1:p
            cdata_sizes = sparse(cols, num_comp_events);
            for n = 1:num_comp_events
                if ~isempty(time_data_idx{n, 2})
                    cdata_sizes(:, n) = sum(ndata_sizes{i}(:, time_data_idx{n,2}), 2);
                    cdata_proc(i, n) = sum(ndata_proc(i, time_data_idx{n,2}), 2);
                end
            end
            cdata_time = cdata_time + cdata_sizes;
            clear cdata_sizes;
        end
    end
    clear time_data_idx;
    clear num_comp_events;
    
    computeTimeStats(fid, ctotals, cdurations, mx, p, labels, hasDataSizes, cdata_time, kernel, scaling, cdata_proc);

    clear cdata_time;
    clear cdata_proc;
    clear ctotals;
    clear cdurations;
    clear labels;
    
    
    clear ntotals;
    clear ndurations;
    clear ndata_sizes;
    clear ndata_proc;
    clear event_names;
    clear event_types;
    clear unique_types;
    clear p;
    clear scaling;
    clear cols;
    clear kernel;
    clear mx;
    clear hasDataSizes;
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
        if length(durations{n}) == 0
            eventmean(n) = 0;
            eventstdev(n) = 0;
            eventmedian(n) = 0;
            eventmode(n) = 0;
            eventmin(n) = 0;
            eventmax(n) = 0;            
        else
            eventmean(n) = mean(durations{n});
            eventstdev(n) = std(double(durations{n}));
            eventmedian(n) = median(double(durations{n}));
            eventmode(n) = mode(double(durations{n}));
            eventmin(n) = min(durations{n});
            eventmax(n) = max(durations{n});
        end
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
        clear data_t;
        
        tp = double(data_proc) ./ double(totals) * 1000000 * scaling;
        TPtotal = sum(tp, 1);
        TPmean = mean(tp, 1);
        TPmedian = median(tp, 1);
        TPmode = mode(tp,1);
        TPstdev = std(tp, 0, 1);
        TPmin = min(tp, [], 1);
        TPmax = max(tp, [], 1);
        clear tp;
        
    end
    clear mx;
    clear mn;
    clear av;
    clear med;
    clear mo;
    
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
        
        clear maxTPIn1sec;
        clear avgTP;
        clear medianTP;
        clear modeTP;
        clear minTP;
        clear maxTP;
        clear TPtotal;
        clear TPmean;
        clear TPstdev;
        clear TPmedian;
        clear TPmode;
        clear TPmin;
        clear TPmax;
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

    clear label_count;
    clear tot;
    clear percent;
    clear eventmean;
    clear eventstdev;
    clear eventmedian;
    clear eventmode;
    clear eventmin;
    clear eventmax;
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
        
        [~, idx] = ismember(names, event_names);
        clear names;
        
        startt_bucket = ceil(startt / sample_interval);
        endt_bucket = ceil(endt / sample_interval);
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;

        duration = endt - startt + 1;  % data rate per microsec.
        
        for j = 1:num_ev_names
            ndata_proc(i, j) = sum(datasize(idx == j),1);
        end

        %datarate = datasize ./ duration;
        fulldr = (datasize * sample_interval) ./ duration;
        % can't do this in more vectorized way, because of the range access
        
        tmpdata = zeros(cols, num_ev_names);
        startdr = datasize .* (start_bucket_end - startt + 1) ./ duration;
        enddr = datasize .* (endt - end_bucket_start + 1) ./ duration;

        clear startt;
        clear endt;

        clear start_bucket_end;
        clear end_bucket_start;

        clear duration;
        
        for j = 1:length(datasize)
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
        clear x1;
        clear x2;
        clear y;
        clear startdr;
        clear enddr;
        clear fulldr;
        clear startt_bucket;
        clear endt_bucket;
    	clear idx;
        clear datasize;

        
        ndata_sizes{i} = sparse(tmpdata);
        clear tmpdata;
    end

    
    ndata_time = sparse(cols, num_ev_names);
    for i = 1:p	
		ndata_time = ndata_time + ndata_sizes{i};
	end
    clear num_ev_names;
end
