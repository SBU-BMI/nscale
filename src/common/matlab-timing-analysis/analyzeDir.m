function analyzeDir ( dirname, allEventTypes, allTypeNames, colorMap, errorfid )
%analyzeDir perform analysis on the csv files inside the specified
%directory.
%   Detailed explanation goes here
close all;

timeInterval = 100000;
procWidth = 1;

fid = fopen([dirname, '.summary.csv'], 'w');
fclose(fid);

files = dir(fullfile(dirname, '*.csv'));

for i = 1:length(files)
    [~, n, ~] = fileparts(files(i).name);
    prefix = fullfile(dirname, n);
    
    try
        clear proc_events;
        
        
        %fprintf(1, 'OLD VERSION\n');
        %proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), proc_types_to_exclude, event_names_to_exclude);
        %fprintf(1, 'OLD VERSION DONE\n');
        
        % read the data in
        tic
        fprintf(1, 'loading data\n');
        datafile = [prefix '.data.mat'];
        if (exist(datafile, 'file') == 2)
            fprintf(1, 'load previously loaded mat data file %s\n', datafile);
            load(datafile);
        else
            fprintf(1, 'parsing %s/%s\n', dirname, files(i).name);
            [proc_events fields] = readLog(dirname, files(i));
            
            if (size(proc_events, 1) == 0)
                fprintf(2, 'No log entry in file %s\n', fullfile(dirname, files(i)));
                continue;
            end
            
            save(datafile, 'proc_events', 'fields');
        end
        toc
        
        % get the min and max timestamps for the total time.
        nrows = size(proc_events,1);
        mxx = zeros(nrows, 1);
        mnx = ones(nrows, 1) * realmax('double');
        et_id = fields.('endT');
        st_id = fields.('startT');
        for p = 1:nrows 
            if (~isempty(proc_events{p, et_id}))
                mxx(p) = max(proc_events{p, et_id}, [], 1);
            end
            if (~isempty(proc_events{p, st_id}))
                mnx(p) = min(proc_events{p, st_id}, [], 1);
            end
        end
        mx = max(mxx, [], 1);  % maximum end timestamp
        mn = min(mnx, [], 1)-1;  % min end timestamp

        
        
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
        
        
        %             %% RENDER
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
        
        
        %% SUMMARIZE
        
        fprintf(1, 'summarizing\n');
        fid = fopen([dirname, '.summary.v2.csv'], 'a');
        fprintf(fid, '%s, start time, %f, end time, %f\n', prefix, mn, mx);        
    
        
        summarize2(proc_events, fields, prefix, fid, allEventTypes, allTypeNames, timeInterval, [mn, mx]);

        
        
        fclose(fid);
     catch err
         fprintf(errorfid, 'ERROR: failed processing for %s, reason: %s\n', prefix, err.message);
     end
end

end

