function analyzeDir ( dirname, allEventTypes, allTypeNames, colorMap, errorfid )
%analyzeDir perform analysis on the csv files inside the specified
%directory.
%   Detailed explanation goes here
close all;

timeInterval = 200000;
procWidth = 1;

%summaryFilename = [dirname, '.summary.v2.walltimes.csv'];
summaryFilename = [dirname, '.summary.v2.csv'];

fid = fopen(summaryFilename, 'w');
fclose(fid);

files = dir(fullfile(dirname, '*.csv'));

for i = 1:length(files)
    [~, n, ~] = fileparts(files(i).name);
    prefix = fullfile(dirname, n);
    
    try
        clear proc_events;
        clear events_pid;
        clear summary_pid;
        clear TPIntervals;
        
        
        %fprintf(1, 'OLD VERSION\n');
        %proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), proc_types_to_exclude, event_names_to_exclude);
        %fprintf(1, 'OLD VERSION DONE\n');
        
        fprintf(1, 'processing %s\n', prefix);
        
        % read the data in
        tic;
        datafile = [prefix '.data.mat'];
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'proc_events'; 'fields'}, who('-file', datafile))) == 2)
            fprintf(1, 'loading previously parsed log data %s\n', datafile);
            load(datafile);
        else
            fprintf(1, 'parsing log data %s/%s\n', dirname, files(i).name);
            [proc_events fields] = readLog(dirname, files(i));
            
            if (size(proc_events, 1) == 0)
                fprintf(2, 'No log entry in file %s\n', fullfile(dirname, files(i)));
                continue;
            end
            
            save(datafile, 'proc_events', 'fields', '-v7.3');
        end
        toc

        % aggregate by pid
        tic;
        datafile = [prefix '.events_pid.mat'];
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'events_pid'; 'fields'; 'nodeTypes'}, who('-file', datafile))) == 3)
            fprintf(1, 'load previously aggregated process log data %s\n', datafile);
            load(datafile);
        else
            fprintf(1, 'aggregating events data based on pid.\n');
            [events_pid nodeTypes] = aggregatePerProc(proc_events, fields);
          
            save(datafile, 'events_pid', 'fields', 'nodeTypes', '-v7.3');
        end
        toc
        
                
        % FIXED THIS IN THE INPUT CSV files.  see fix_na-POSIX_logs.sh in scripts
        %            if (~isempty(strfind(files(i).name, 'na-POSIX')))
        %                %tic;
        %                %fprintf(1, 'checking %s to have na-POSIX write size\n', files(i).name);
        %                for k = 1:size(proc_events, 1)
        %                    idx = find(strcmp('IO POSIX Write', proc_events{k, fields.('eventName')}), 1);
        %                    if (~isempty(idx))
        %                        if (proc_events{k, fields.('attribute')}(idx) == 0)
        %                            fprintf(2, 'missing na-POSIX write size in %s\n', files(i).name);
        %                           %proc_events{k, 8}(idx) = 67109117;
        %                        end
        %                    end
        %                end
        %                %toc
        %            end
        
        
%% RENDER
        %             % filter out MPI messaging
        %             tic
        %             fprintf(1, 'prepare to render - exclude MPI send/receive\n');
        %             eventFilter = cell(1,2);
        %             eventFilter{1,1} = '~eventType';
        %             eventFilter{1,2} = 21;
        %             temp_events = filterEvents(proc_events, fields, eventFilter, 'or');
        %             toc
        %
        %
        %             tic;
        %             fprintf(1, 'render - exclude MPI send/receive\n');
        %             [~, ~, ~] = plotProcEvents(temp_events, procWidth, timeInterval, prefix, allEventTypes, colorMap);
        %             close all;
        %             clear temp_events;
        %             toc
        %             %save([dirname filesep files(i).name '.mat'], 'norm_events', 'sum_events');
        

        %% compute intermediate results
        % compute min and max
        tic;
        fprintf(1, 'compute min and max\n');
        nrows = size(events_pid,1);
        mxx = ones(nrows, 1) * -1.0;
        mnx = ones(nrows, 1) * -1.0;
        et_id = fields.('endT');
        st_id = fields.('startT');
        for p = 1:nrows 
            if (~isempty(events_pid{p, et_id}))
                mxx(p) = max(events_pid{p, et_id}, [], 1);
            end
            if (~isempty(events_pid{p, st_id}))
                mnx(p) = min(events_pid{p, st_id}, [], 1);
            end
        end
        mx = max(mxx(mxx>=0), [], 1);  % maximum end timestamp
        mn = min(mnx(mnx>=0), [], 1)-1;  % min end timestamp
        durs = mxx - mnx;
        toc
          
        % intermediate summary by procs
        tic;
        datafile = [prefix '.events_stats.mat'];
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'summary_pid'; 'ops_pid'; 'allEventTypes'}, who('-file', datafile))) == 3)
            fprintf(1, 'load previously computed intermediate summary %s\n', datafile);
            load(datafile);
        else
            fprintf(1, 'compute intermediate summary for events data based on pid.\n');
            [summary_pid ops_pid] = summarizeByRow(events_pid, fields, allEventTypes);
            save(datafile1, 'summary_pid', 'ops_pid', 'allEventTypes', '-v7.3');
        end
        toc
        
        
        % resample datasizes by events
        tic;
        datafile = [prefix '.events_resample.mat'];
       	if (exist(datafile, 'file') == 2 &&...
                length(intersect({'TPIntervals', 'interval', 'allEventTypes'}, who('-file', datafile))) == 3)
            fprintf(1, 'load previously saved resampled data file %s\n', datafile);
            load(datafile);
        else
            fprintf(1, 'resampling data by events\n');
            [TPIntervals, interval, ~] = resampleData(events_pid, fields, timeInterval, allEventTypes, 'eventType', [mn mx]);
            save(datafile, 'TPIntervals', 'interval', 'allEventTypes', '-v7.3');       
        end
        toc

        

        
        
%% SUMMARIZE
        
        fprintf(1, 'summarizing\n');
        fid = fopen(summaryFilename, 'a');
        fprintf(fid, 'EXPERIMENT, %s, app wall time, %f, sum process wall time, %f\n', prefix, mx-mn, sum(durs));        
    
        tic;
        summarizeFromIntermeidate(summary_pid, ops_pid, TPIntervals, interval, allEventTypes, allTypeNames, fid);
        toc
        
        
        fclose(fid);
     catch err
        fprintf(errorfid, 'ERROR: failed processing for %s, reason: %s\n', prefix, err.message);
     end
end

end

