function analyzeDir ( dirname, allEventTypes, allTypeNames, colorMap )
%analyzeDir perform analysis on the csv files inside the specified
%directory.
%   Detailed explanation goes here
    close all;

    timeInterval = 100000;
    procWidth = 1;
    proc_types_to_exclude = {'m'};
    event_names_to_exclude = {'MPI NB SEND', 'MPI NB RECV', 'MPI B SEND', 'MPI B RECV'};


    fid = fopen([dirname, '.summary.csv'], 'w');
    fclose(fid);

    files = dir(fullfile(dirname, '*.csv'));

    for i = 1:length(files)
         try
            clear proc_events;

            proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), proc_types_to_exclude, event_names_to_exclude);

% FIX THIS IN THE INPUT CSV files instead...
            tic;
            if (~isempty(strfind(files(i).name, 'na-POSIX')))
                fprintf(1, 'checking %s to have na-POSIX write size\n', files(i).name);
                for k = 1:size(proc_events, 1)
                    idx = strcmp('IO POSIX Write', proc_events{k, 4});
                    if (length(idx) > 0)
                        idx1 = idx(1);
                        if (proc_events{k, 8}(idx1) == 0)
                            fprintf(1, 'missing na-POSIX write size in %s\n', files(i).name);
                           %proc_events{k, 8}(idx) = 67109117;
                        end 
                    end
                end
                fprintf(1, 'finished checking %s to have na-POSIX write size\n', files(i).name);

            end
            toc

            [~, n, ~] = fileparts(files(i).name);
            prefix = fullfile(dirname, n);
            [~, norm_events, sum_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix, allEventTypes, colorMap);
            save([dirname filesep files(i).name '.mat'], 'norm_events', 'sum_events');            
            close all;

            fid = fopen([dirname, '.summary.csv'], 'a');
            fprintf(fid, '%s\n', prefix);
            tic;
            fprintf(1, 'summarizing\n');
            summarize(proc_events, timeInterval, fid, '*', allEventTypes, allTypeNames);
            toc
            fclose(fid);
        catch err
            fprintf(2, 'ERROR: failed processing for %s, reason: %s\n', files(i).name, err.message);
        end
    end

end

