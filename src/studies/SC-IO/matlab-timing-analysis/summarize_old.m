function [] = summarize_old( proc_events, sample_interval, fid, proc_type, allEventTypes, allTypeNames)
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
    pidx = strcmp(proc_type, cat(1, proc_events{:, 3}));
    mx = zeros(1, size(proc_events, 1));
    for i = 1:size(proc_events, 1) 
        mx(i) = max(proc_events{i, 7});  % maximum end timestamp
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

    
    %% first analyze by event name

    % slow version with dataset and unique.
    tic;
    p = size(events, 1);
    % get the unique event names, type mapping, and also the count
    unique_events = dataset();
    for i = 1:p
        % get the unique set.  uses a dataset to ensure that name and type
        % together form a unique set.
        unique_events = unique([unique_events;...
            dataset(events{i, 4}, events{i, 5})], ...
            1);
    end
    event_names1 = unique_events.Var1;
    event_types1 = unique_events.Var2;
    unique_types1 = unique(event_types1);
    num_ev_names = size(event_names1, 1);
    num_ev_types = size(unique_types1, 1);
    clear unique_events;
    t1 = toc;  % 2.1 sec on tcga AMR
    
    % fast version
    tic;
    p = size(events, 1);
    % get the unique event names, type mapping, and also the count
    event_names = {};
    event_types = [];
    for i = 1:p
        event_names = [event_names; events{i,4}];
        event_types = [event_types; events{i,5}];
        [event_names, ia, ~] = unique(event_names);
        event_types = event_types(ia);
    end
    clear ia;
    
    unique_types = unique(event_types);
    num_ev_names = size(event_names, 1);
    num_ev_types = size(unique_types, 1);
    % 0.44 sec on tcgaAMR.
    t2 = toc;
    fprintf(1, 'event names and types.  slow = %f, fast = %f\n', t1, t2);

    
    if ~isempty(find(strcmp(event_names1, event_names) == 0, 1))
        fprintf(2, 'ERROR:  event names not same\n');
        return;
    end
    if ~isempty(find(event_types1 ~= event_types, 1))
        fprintf(2, 'ERROR:  event types not same\n');
        return;
    end
    if ~isempty(find(unique_types1 ~= unique_types, 1))
        fprintf(2, 'ERROR:  unique types not same\n');
        return;
    end
    
    
    % SLOW: get the total durations of each event by name
    tic;
    ntotals1 = zeros(p, num_ev_names);
    ndurations1 = cell(num_ev_names, 1);
    for i = 1:p
        names = events{i, 4};
        % find the unique names and their mapping back into the original
        % list
        [unames, ~, names2unamesMap] = unique(names);
        % now find the mapping of the unique names to the overall unique
        % event names
        [mnames mnames2unamesMap mnames2ueventMap] = intersect(unames, event_names);
        duration = events{i, 7} - events{i, 6} + 1;
        for n = 1:length(mnames)
            % compute total
            npos = mnames2ueventMap(n);
            idx = names2unamesMap == mnames2unamesMap(n);
            tmpdur = duration(idx);
%            idx = find(strcmp(event_names(n), names) == 1);
            ntotals1(i, npos) = sum(tmpdur,1);
            ndurations1{npos} = [ndurations1{npos}; tmpdur];
            clear npos;
            clear idx;
            clear tmpdur;
        end
        clear unames;
        clear mnames;
        clear names2unamesMap;
        clear mnames2unamesMap;
        clear mnames2ueventMap;
        clear duration;
        clear names;
    end
    t1 = toc; 
    
    % ---- fast version
    % get the total durations of each event by name
    % using find(strcmp()) in an inner loop for the names is faster - 0.95s
    % vs 1.7 sec if using unique and intersect to get the mappings.
    tic;
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
        clear duration;
        clear names;
    end
    
    t2 = toc;
    fprintf(1, 'ntotals and ndurations.  slow = %f, fast = %f\n', t1, t2);
    
    if ~isempty(find(ntotals ~= ntotals1, 1))
        fprintf(2, 'ERROR:  ntotals not same\n');
        return;
    end
    for n = 1:num_ev_names
        if ~isempty(find(ndurations{n} ~= ndurations1{n}, 1))
            fprintf(2, 'ERROR:  ndurations %d not same\n', n);
            return;
        end
    end
    
    
    %% then analyze by event type
    
    
    
    
    
    
    
    %% finally analyze by user specified grouping of events
    
    
        

    
    
   
    ntotal = sum(ntotals, 1);
    nmax = max(ntotals, [], 1);
    nmin = min(ntotals, [], 1);
    navg = mean(ntotals, 1);
    nmedian = median(ntotals, 1);
    nmode = mode(ntotals, 1);

    npercent = ntotal / double(mx * p);
    neventmean = zeros(num_ev_names,1);
    neventstdev = zeros(num_ev_names,1);
    neventmedian = zeros(num_ev_names,1);
    neventmode = zeros(num_ev_names,1);
    neventmin = zeros(num_ev_names,1);
    neventmax = zeros(num_ev_names,1);
    for n = 1:num_ev_names
        neventmean(n) = mean(ndurations{n});
        neventstdev(n) = std(double(ndurations{n}));
        neventmedian(n) = median(double(ndurations{n}));
        neventmode(n) = mode(double(ndurations{n}));
        neventmin(n) = min(ndurations{n});
        neventmax(n) = max(ndurations{n});
    end
    
    % ge tthe total durations of each event by type
    ttotals = zeros(p, num_ev_types);
    tdurations = cell(num_ev_types, 1);
    for n = 1:num_ev_names
        % compute total
        id = find(unique_types == event_types(n));
        ttotals(:, id) = ttotals(:, id) + ntotals(:, n);
        tdurations{id} = [tdurations{id}; ndurations{n}];
    end
    ttotal = sum(ttotals, 1);
    tmax = max(ttotals, [], 1);
    tmin = min(ttotals, [], 1);
    tavg = mean(ttotals, 1);
    tmedian = median(ttotals, 1);
    tmode = mode(ttotals, 1);
    
    tpercent = ttotal / double(mx * p);
    teventmean = zeros(num_ev_types,1);
    teventstdev = zeros(num_ev_types,1);
    teventmedian = zeros(num_ev_types,1);
    teventmode = zeros(num_ev_types,1);
    teventmin = zeros(num_ev_types,1);
    teventmax = zeros(num_ev_types,1);
    for t = 1:num_ev_types
        teventmean(t) = mean(tdurations{t});
        teventstdev(t) = std(double(tdurations{t}));
        teventmedian(t) = median(double(tdurations{t}));
        teventmode(t) = mode(double(tdurations{t}));
        teventmin(t) = min(tdurations{t});
        teventmax(t) = max(tdurations{t});
    end

    
    % compute throughput
    nmaxTPIn1sec = zeros(num_ev_names, 1);
    %navgTPAvgNode = zeros(1, num_ev_names);
    navgTP = zeros(1, num_ev_names);
    nminTP = zeros(1, num_ev_names);
    nmaxTP = zeros(1, num_ev_names);
    nmedianTP = zeros(1, num_ev_names);
    nmodeTP = zeros(1, num_ev_names);
    
    nTPtotal = zeros(1, num_ev_names);
    nTPmean = zeros(1, num_ev_names);
    nTPmedian = zeros(1, num_ev_names);
    nTPmode = zeros(1, num_ev_names);
    nTPstdev = zeros(1, num_ev_names);
    nTPmin = zeros(1, num_ev_names);
    nTPmax = zeros(1, num_ev_names);
    
    tmaxTPIn1sec = zeros(num_ev_types, 1);
    %tavgTPAvgNode = zeros(1, num_ev_types);
    tavgTP = zeros(1, num_ev_types);
    tminTP = zeros(1, num_ev_types);
    tmaxTP = zeros(1, num_ev_types);
    tmedianTP = zeros(1, num_ev_types);
    tmodeTP = zeros(1, num_ev_types);

    tTPtotal = zeros(1, num_ev_types);
    tTPmean = zeros(1, num_ev_types);
    tTPmedian = zeros(1, num_ev_types);
    tTPmode = zeros(1, num_ev_types);
    tTPstdev = zeros(1, num_ev_types);
    tTPmin = zeros(1, num_ev_types);
    tTPmax = zeros(1, num_ev_types);
    
    % throughputs in gigabytes
    scaling = 1 / double(1024*1024*1024); % GB/s
    if (size(events, 2) > 7)
        cols = ceil(double(mx) / double(sample_interval));

        [ndata_time1 ndata_proc1 ~] = resampleDataByName(events, p, cols, sample_interval, event_names);
        
        [ndata_time tdata_time ndata_proc tdata_proc] = resampleEvents(events, p, mx, sample_interval, event_names, event_types);
        
        if ~isempty(find(ndata_time ~= ndata_time1, 1))
            fprintf(2, 'ERROR: ndata_time not same\n');
        end
        if ~isempty(find(ndata_proc ~= ndata_proc1, 1))
            fprintf(2, 'ERROR: ndata_proc not same\n');
        end
        clear ndata_time1;
        clear ndata_proc1;
        
        
        % find the 1 sec interval during whick we have max IO.
        kernel = ones(round(1000000 / sample_interval), 1, 'double');
        
        % maxThroughput in 1 sec is the maximum running average
        for n = 1:num_ev_names
            nmaxTPIn1sec(n) = max(conv(full(ndata_time(:,n)), kernel, 'same')) * scaling;
        end
        ndata_t = sum(ndata_proc, 1);
        % avg TP per node is total data / total CPU time.
        %navgTPAvgNode = ndata_t ./ ntotal * 1000000 * scaling ;
        % avg TP is total data / average CPU time
        navgTP = ndata_t ./ navg * 1000000 * scaling ;
        nminTP = ndata_t ./ nmax * 1000000 * scaling ;
        nmaxTP = ndata_t ./ nmin * 1000000 * scaling ;
        nmedianTP = ndata_t ./ nmedian * 1000000 * scaling ;
        nmodeTP = ndata_t ./ nmode * 1000000 * scaling ;
        
        tp = double(ndata_proc) ./ double(ntotals) * 1000000 * scaling;
        nTPtotal = sum(tp, 1);
        nTPmean = mean(tp, 1);
        nTPmedian = median(tp, 1);
        nTPmode = mode(tp,1);
        nTPstdev = std(tp, 1);
        nTPmin = min(tp, [], 1);
        nTPmax = max(tp, [], 1);
        clear tp;
        
        for t = 1:num_ev_types
            tmaxTPIn1sec(t) = max(conv(full(tdata_time(:,t)), kernel, 'same')) * scaling;
        end
        tdata_t = sum(tdata_proc, 1);
        %tavgTPAvgNode = sum(tdata_proc, 1) ./ ttotal * 1000000 * scaling;
        tavgTP = tdata_t ./ tavg * 1000000 * scaling ;
        tminTP = tdata_t ./ tmax * 1000000 * scaling ;
        tmaxTP = tdata_t ./ tmin * 1000000 * scaling ;
        tmedianTP = tdata_t ./ tmedian * 1000000 * scaling ;
        tmodeTP = tdata_t ./ tmode * 1000000 * scaling ;
        
        tp = double(tdata_proc) ./ double(ttotals) * 1000000 * scaling;
        tTPtotal = sum(tp, 1);
        tTPmean = mean(tp, 1);
        tTPmedian = median(tp, 1);
        tTPmode = mode(tp,1);
        tTPstdev = std(tp, 1);
        tTPmin = min(tp, [], 1);
        tTPmax = max(tp, [], 1);
        clear tp;
        

	end
    fprintf(fid, 'Operation,TotalTime(ms),PercentTime,MeanTime,StdevTime,MedianTime,ModeTime,MinTime,MaxTime,MaxTPin1sec(GB/s),avgTP=sum(data)/mean(t),medianTP,modeTP,minTP,maxTP,nodeTPtotal=sum(data/t),nodeTPavg,nodeTPstdev,nodeTPmedian,nodeTPmode,nodeTPmin,nodeTPmax\n');
    for n = 1:num_ev_names
        fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
            event_names{n}, ntotal(n)/1000, npercent(n), ...
            neventmean(n)/1000, neventstdev(n)/1000, ...
            neventmedian(n)/1000, neventmode(n)/1000,...
            neventmin(n)/1000, neventmax(n)/1000,...
            nmaxTPIn1sec(n), ...
            navgTP(n), nmedianTP(n), nmodeTP(n),...
            nminTP(n), nmaxTP(n), ...
            nTPtotal(n),nTPmean(n),nTPstdev(n),nTPmedian(n),nTPmode(n),...
            nTPmin(n),nTPmax(n));
    end
    fprintf(fid, '\n');
    fprintf(fid, 'Operation Type,TotalTime(ms),PercentTime,MeanTime(ms),StdevTime(ms),MedianTime,ModeTime,MinTime,MaxTime,MaxTPin1sec(GB/s),avgTP=sum(data)/mean(t),medianTP,modeTP,minTP,maxTP,nodeTPtotal=sum(data/t),nodeTPavg,nodeTPstdev,nodeTPmedian,nodeTPmode,nodeTPmin,nodeTPmax\n');
    for t = 1:num_ev_types
        fprintf(fid, '%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
            allTypeNames{allEventTypes == unique_types(t)},...
            ttotal(t)/1000, tpercent(t), teventmean(t)/1000, teventstdev(t)/1000, ...
            teventmedian(t)/1000, teventmode(t)/1000,...
            teventmin(t)/1000, teventmax(t)/1000,...
            tmaxTPIn1sec(t), ...
            tavgTP(t), tmedianTP(t), tmodeTP(t), ...
            tminTP(t), tmaxTP(t), ...
            tTPtotal(t),tTPmean(t),tTPstdev(t),tTPmedian(t),tTPmode(t),...
            tTPmin(t),tTPmax(t));
    end
    fprintf(fid, '\n');
    

    % application specific computation.
    num_tiles = zeros(p, 1);
    for i = 1:p
       num_tiles(i) = sum(events{i, 5} == 0, 1); 
    end
    
    % compute the write throughput : 
    % need to know how many tiles were processed.
    % first do per node.
    
    if (size(events, 2) > 7)

        % preparation:
        data_cols = [find(strcmp('IO define vars', event_names)),...
            find(strcmp('IO tile data', event_names)), ...
            find(strcmp('IO MPI scan', event_names)), ...
            find(strcmp('adios close', event_names))];
        prep_cols = [find(strcmp('IO define vars', event_names)),...
            find(strcmp('IO malloc vars', event_names)), ...
            find(strcmp('IO tile metadata', event_names)), ...
            find(strcmp('IO tile data', event_names)), ...
            find(strcmp('IO MPI scan', event_names)), ...
            find(strcmp('IO MPI allreduce', event_names)), ...
            find(strcmp('IO var clear', event_names))];
        adios_cols = [find(strcmp('adios open', event_names)),...
            find(strcmp('ADIOS WRITE', event_names)),...
            find(strcmp('ADIOS WRITE Summary', event_names)), ...
            find(strcmp('adios close', event_names))];
        
        if (~isempty(data_cols) && ~isempty(prep_cols) && ~isempty(adios_cols))
            
            % prep time
            prep_data_proc = ndata_proc(:, data_cols(2));  % data written out
            prep_time_proc = sum(ntotals(:, prep_cols), 2);

            computeThroughput('AdiosPrep', prep_data_proc, prep_time_proc, 1000000 * scaling, fid);

            % adios
            adios_data_proc = ndata_proc(:, data_cols(4));
            adios_time_proc = sum(ntotals(:, adios_cols), 2);

            computeThroughput('Adios', adios_data_proc, adios_time_proc, 1000000 * scaling, fid);
 
        end
        
    end
        clear ndata_time;
        clear tdata_time;
        clear ndata_proc;
        clear tdata_proc;

end

function computeThroughput(name, data_proc, time_proc, scaling, fid)
    tp = double(data_proc) ./ double(time_proc) * scaling;
    TPtotal = sum(tp, 1);
    TPmean = mean(tp, 1);
    TPmedian = median(tp, 1);
    TPmode = mode(tp,1);
    TPstdev = std(tp, 1);
    TPmin = min(tp, [], 1);
    TPmax = max(tp, [], 1);
    clear tp;

    data_t = sum(data_proc,1) * scaling;
    avgTP = data_t ./ mean(time_proc, 1);
    minTP = data_t ./ max(time_proc, [], 1);
    maxTP = data_t ./ min(time_proc, [], 1);
    medianTP = data_t ./ median(time_proc, 1);
    modeTP = data_t ./ mode(time_proc, 1);
    clear data_t;

    fprintf(fid, 'Operation,TotalTime(ms),PercentTime,MeanTime,StdevTime,MedianTime,ModeTime,MinTime,MaxTime,MaxTPin1sec(GB/s),avgTP=sum(data)/avg(t),medianTP,modeTP,minTP,maxTP,nodeTPtotal=sum(data/t),nodeTPavg,nodeTPstdev,nodeTPmedian,nodeTPmode,nodeTPmin,nodeTPmax\n');
    fprintf(fid, '%s,,,,,,,,,,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n\n', ...
        name, ...
        avgTP, medianTP, modeTP,...
        minTP, maxTP, ...
        TPtotal,TPmean,TPstdev,TPmedian,...
        TPmode, TPmin,TPmax);

end

%% slow
function [ ndata_time tdata_time ndata_proc tdata_proc ] = ...
    resampleEvents(events, p, mx, sample_interval, event_names, event_types)
    % p is number of cells (procs)
    % mx is maximum timestamp overall;

    % get the number of pixels
    cols = ceil(double(mx) / double(sample_interval));
    num_ev_names = length(event_names);
    unique_types = unique(event_types);
    num_ev_types = length(unique_types);
    % allocate norm-events
	ndata_sizes = cell(p, 1);
    ndata_proc = zeros(p, num_ev_names);
    
    % generate the sampled_events
    for i = 1:p

        ndata_sizes{i} = sparse(cols, num_ev_names);
        
        names = events{i, 4};
        
        [~, idx] = ismember(names, event_names);
        clear blah;
        startt = double(events{i, 6});
        endt = double(events{i, 7});
        datasize = double(events{i, 8});
        
        startt_bucket = ceil(double(startt) / double(sample_interval));
        endt_bucket = ceil(double(endt) / double(sample_interval));
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;

        duration = double(endt - startt + 1);  % data rate per microsec.
        
        for j = 1:num_ev_names
            ndata_proc(i, j) = sum(datasize(idx == j));
        end

        % can't do this in more vectorized way, because of the range access
        for j = 1:length(names)
            
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
               ndata_sizes{i}(startt_bucket(j), idx(j)) = ...
                   ndata_sizes{i}(startt_bucket(j), idx(j)) + datasize(j);
           else
               % do the start first
               t = start_bucket_end(j) - startt(j) + 1;
               ndata_sizes{i}(startt_bucket(j), idx(j)) = ...
                   ndata_sizes{i}(startt_bucket(j), idx(j)) + datasize(j) * double(t) / duration(j);
                % then do the end
               t = endt(j) - end_bucket_start(j) + 1;
               ndata_sizes{i}(endt_bucket(j), idx(j)) = ...
                   ndata_sizes{i}(endt_bucket(j), idx(j)) + datasize(j) * double(t) / duration(j);

               % then do in between
               if endt_bucket(j) > (startt_bucket(j) + 1)
                    ndata_sizes{i}(startt_bucket(j)+1 : endt_bucket(j)-1, idx(j)) = ...
                       ndata_sizes{i}(startt_bucket(j)+1 : endt_bucket(j)-1, idx(j)) + datasize(j) * double(sample_interval) / duration(j);
               end
           end 
        end
        
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

    % aggregate to ge tdata sizes
    tdata_sizes = cell(p, 1);
    tdata_proc = zeros(p, num_ev_types);
    for i = 1:p
        tdata_sizes{i} = sparse(cols, num_ev_types);
        for n = 1:num_ev_names
            id = find(unique_types == event_types(n));
            tdata_sizes{i}(:, id) = tdata_sizes{i}(:, id) + ndata_sizes{i}(:, n);
            tdata_proc(i, id) = tdata_proc(i, id) + ndata_proc(i, n);
        end
    end
    clear id;
    
    ndata_time = sparse(cols, num_ev_names);
    tdata_time = sparse(cols, num_ev_types);
	for i = 1:p	
		ndata_time = ndata_time + ndata_sizes{i};
		tdata_time = tdata_time + tdata_sizes{i};
    end

    clear unique_types;
    clear tdata_sizes;
    clear ndata_sizes;
end

%% faster
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
    t1 = 0;
    t2 = 0;
    t3 = 0;
    for i = 1:p

        ndata_sizes{i} = sparse(cols, num_ev_names);
        
        names = events{i, 4};
        startt = double(events{i, 6});
        endt = double(events{i, 7});
        datasize = double(events{i, 8});
        
        [~, idx] = ismember(names, event_names);
        
        startt_bucket = ceil(startt / sample_interval);
        endt_bucket = ceil(endt / sample_interval);
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;
        start_bucket_start = (startt_bucket - 1) * sample_interval + 1;
        end_bucket_end = endt_bucket * sample_interval;

        duration = endt - startt + 1;  % data rate per microsec.
        
        for j = 1:num_ev_names
            ndata_proc(i, j) = sum(datasize(idx == j),1);
        end

        %datarate = datasize ./ duration;
        %fulldr = datarate * double(sample_interval);
        fulldr = (datasize * sample_interval) ./ duration;
        % can't do this in more vectorized way, because of the range access
        



        
        tmpdata = zeros(cols, num_ev_names);
        tic;
        % no plus 1 at the end because start and endt are not to be
        % subtracted
        startdr2 = datasize .* (startt - start_bucket_start) ./ duration;
        enddr2 = datasize .* (end_bucket_end - endt) ./ duration;
        for j = 1:length(names)
            x1 = startt_bucket(j);
            x2 = endt_bucket(j);
            y = idx(j);
            
           % if start and end in the same bucket, mark with duration
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
        t2 = t2 + toc;
        
        
        tmpdata = zeros(cols, num_ev_names);
        tic;
        % no plus 1 at the end because start and endt are not to be
        % subtracted
        startdr2 = datasize .* (startt - start_bucket_start) ./ duration;
        enddr2 = datasize .* (end_bucket_end - endt) ./ duration;
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
        t3 = t3 + toc;

                tmpdata = zeros(cols, num_ev_names);
        tic;
        %startdr = datarate .* double(start_bucket_end - startt + 1);
        %enddr = datarate .* double(endt - end_bucket_start + 1);
        startdr = datasize .* (start_bucket_end - startt + 1) ./ duration;
        enddr = datasize .* (endt - end_bucket_start + 1) ./ duration;
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
        t1 = t1 + toc;
        
        
        
        
        ndata_sizes{i} = sparse(tmpdata);
    	clear idx;
        clear names;
        clear startt;
        clear endt;
        clear startt_bucket;
        clear endt_bucket;
        clear start_bucket_end;
        clear end_bucket_start;
        clear start_bucket_start;
        clear end_bucket_end;
        clear datasize;
        clear duration;
        clear tmpdata;
    end
    fprintf(1, 'sampling.  lots conditional (numerically better?) = %f, no conditional = %f, some conditional = %f\n', t1, t2, t3);

    
    ndata_time = sparse(cols, num_ev_names);
    for i = 1:p	
		ndata_time = ndata_time + ndata_sizes{i};
	end
    clear num_ev_names;
end
