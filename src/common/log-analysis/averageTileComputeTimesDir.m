function averageTileComputeTimesDir ( dirname, errorfid )
%averageTileComputeTimesDir perform get the average tile compute times
%directory.
%  computes the average, min, max, and mean computation time for tiles of different types
%  types are determined via the event names of computeFull, computeNoNU, computeNoFG. 
close all;

timeInterval = 200000;

%summaryFilename = [dirname, '.summary.v2.walltimes.csv'];
summaryFilename = [dirname, '.tileComputeTimes.csv'];

fid = fopen(summaryFilename, 'w');
fclose(fid);

files = dir(fullfile(dirname, '*.csv'));

for i = 1:length(files)
    [~, n, ~] = fileparts(files(i).name);
    prefix = fullfile(dirname, n);
    
        clear proc_events;
        clear events_pid;

        
        %fprintf(1, 'OLD VERSION\n');
        %proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), proc_types_to_exclude, event_names_to_exclude);
        %fprintf(1, 'OLD VERSION DONE\n');
        
        fprintf(1, 'processing %s\n', prefix);
        

        % aggregate by pid
        tic;
        datafile = [prefix '.events_pid.mat'];
        flag = exist(datafile, 'file') == 2 &&...
                length(intersect({'events_pid'; 'fields'; 'nodeTypes'}, who('-file', datafile))) == 3;
        toc
        if (flag)
            tic;
            fprintf(1, 'load previously aggregated process log data %s\n', datafile);
            load(datafile);
            toc
        else
            % read the data in
            tic;
            datafile2 = [prefix '.data.mat'];
            if (exist(datafile2, 'file') == 2 &&...
                    length(intersect({'proc_events'; 'fields'}, who('-file', datafile2))) == 2)
                fprintf(1, 'loading previously parsed log data %s\n', datafile2);
                load(datafile2);
            else
                fprintf(1, 'parsing log data %s/%s\n', dirname, files(i).name);
                [proc_events fields] = readLog(dirname, files(i));

                if (size(proc_events, 1) == 0)
                    fprintf(2, 'No log entry in file %s\n', fullfile(dirname, files(i)));
                    continue;
                end

                save(datafile2, 'proc_events', 'fields');
            end
            toc
            
            tic;
            fprintf(1, 'aggregating events data based on pid.\n');
            [events_pid nodeTypes] = aggregatePerProc(proc_events, fields);
          
            save(datafile, 'events_pid', 'fields', 'nodeTypes');
            toc
        end
        
                

        
%% SUMMARIZE
        
        fprintf(1, 'summarizing\n');
        fid = fopen(summaryFilename, 'a');
	% wall time reported in microsec.  fixed in perl extraction script.
        %fprintf(fid, 'EXPERIMENT, %s, app wall time, %f, sum process wall time, %f\n', prefix, mx-mn, sum(durs));        
    
        tic;

        eventnames = cat(1, events_pid{:, fields.('eventName')});
        computeFullIdx = strcmp('computeFull', eventnames);
        
        
        computeNoNUIdx = strcmp('computeNoNU', eventnames);
        computeNoFGIdx = strcmp('computeNoFG', eventnames);
        startimes = cat(1, events_pid{:, fields.('startT')});
        endtimes = cat(1, events_pid{:, fields.('endT')});
        
        fullDuration = double(endtimes(computeFullIdx) - startimes(computeFullIdx) + 1);
        noNUDuration = double(endtimes(computeNoNUIdx) - startimes(computeNoNUIdx) + 1);
        noFGDuration = double(endtimes(computeNoFGIdx) - startimes(computeNoFGIdx) + 1);
        
        totalcount = length(fullDuration) + length(noNUDuration) + length(noFGDuration);
        percentFull = length(fullDuration) / totalcount;
        percentNoNU = length(noNUDuration) / totalcount;
        percentNoFG = length(noFGDuration) / totalcount;
        
        meanFull = mean(fullDuration);
        meanNoNU = mean(noNUDuration);
        meanNoFG = mean(noFGDuration);
        stdFull = std(fullDuration);
        stdNoNU = std(noNUDuration);
        stdNoFG = std(noFGDuration);
        
        fprintf(fid, '%s,%d: %f,%f,%f;%f,%f,%f;%f,%f,%f\n', prefix, totalcount, ...
            percentNoFG, meanNoFG, stdNoFG, ...
            percentNoNU, meanNoNU, stdNoNU, ...
            percentFull, meanFull, stdFull);
        
        toc
        
        fclose(fid);

end

end

