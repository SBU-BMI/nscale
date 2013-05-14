function checkDir ( dirname, allEventTypes, allTypeNames, colorMap, errorfid )
%checkDir check to see if we have missing intermediate files
%   Detailed explanation goes here
close all;

timeInterval = 200000;
procWidth = 1;

files = dir(fullfile(dirname, '*.csv'));

for i = 1:length(files)
    [~, n, ~] = fileparts(files(i).name);
    prefix = fullfile(dirname, n);
    
%    try
        clear proc_events;
        clear events_pid;
        clear summary_pid;
        clear TPIntervals;
        
        
        %fprintf(1, 'OLD VERSION\n');
        %proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), proc_types_to_exclude, event_names_to_exclude);
        %fprintf(1, 'OLD VERSION DONE\n');
        
        %fprintf(1, 'processing %s\n', prefix);
        fprintf(1, '.');
        if (mod(i, 40) == 0)
            fprintf(1, '\n');
        end
        
        % read the data in
        datafile = [prefix '.data.mat'];
        if (getmatv(datafile))
            fprintf(1, 'v7.3 file: %s\n', datafile);
        end
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'proc_events'; 'fields'}, who('-file', datafile))) == 2)
        else
            fprintf(1, '\n need to parse log data %s\n', datafile);
        end

        % aggregate by pid
        datafile = [prefix '.events_pid.mat'];
        if (getmatv(datafile))
            fprintf(1, 'v7.3 file: %s\n', datafile);
        end
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'events_pid'; 'fields'; 'nodeTypes'}, who('-file', datafile))) == 3)
        else
            fprintf(1, '\n need aggregating events data based on pid. %s\n', datafile);
        end
        
          
        % intermediate summary by procs
        datafile = [prefix '.events_stats.mat'];
        if (getmatv(datafile))
            fprintf(1, 'v7.3 file: %s\n', datafile);
        end
        if (exist(datafile, 'file') == 2 &&...
                length(intersect({'summary_pid'; 'ops_pid'; 'allEventTypes'}, who('-file', datafile))) == 3)
        else
            fprintf(1, '\n need compute intermediate summary for events data based on pid. %s\n', datafile);
        end
        
        
        % resample datasizes by events
        datafile = [prefix '.events_resample.mat'];
        if (getmatv(datafile))
            fprintf(1, 'v7.3 file: %s\n', datafile);
        end
       	if (exist(datafile, 'file') == 2 &&...
                length(intersect({'TPIntervals', 'interval', 'allEventTypes'}, who('-file', datafile))) == 3)
        else
            fprintf(1, '\n need resampling data by events.  %s\n', datafile);
        end

        
%      catch err
%         fprintf(errorfid, 'ERROR: failed processing for %s, reason: %s\n', prefix, err.message);
%      end
end

end

function tf = getmatv(fname)
	x = evalc(['type(''', fname, ''')']);
	tf = strcmp(x(2:20), 'MATLAB 7.3 MAT-file');
end

