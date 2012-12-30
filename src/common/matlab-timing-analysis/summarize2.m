function summarize2(proc_events, fields, prefix, fid, allEventTypes, allTypeNames, timeInterval, timeRange)

	datafile = [prefix '.summary.v2.mat'];
        if (exist(datafile, 'file') == 2)
            fprintf(1, 'load previously saved mat data file %s\n', datafile);
            load(datafile);
        else
	
            
        tic
        fprintf(1, 'aggregate events into pid based and relabel sessions\n');
        % group by pid first
        events_pid = aggregateProcesses(proc_events, fields, {'pid'});
        % now replace the sessionNames.
        sessions = unique(cat(1, proc_events{:, fields.('sessionName')}));
        % next look for the right set
        if (length(intersect(sessions, {'m', 'w'})) == 2)
            nodeTypes = {'m', 'w'};
        elseif (length(intersect(sessions, {'assign', 'io', 'seg'})) == 3)
            nodeTypes = {'assign', 'io', 'seg'};
        elseif (length(intersect(sessions, {'io', 'seg'})) == 2)
            nodeTypes = {'io', 'seg'};
        else
            fprintf(2, 'ERROR: log does not have appropriate node session types\n  ');
            for s = 1:length(sessions)
                fprintf(2, '%s,', sessions{s});
            end
            fprintf(2, '\n');
            return;
        end
        
 % THIS CODE IS STUPID SLOW.  USE ALTERNATIVE BELOW.       
 %       events_pid(:, fields.('sessionName')) = ...
 %           cellfun(@(x) intersect(nodeTypes, x), events_pid(:, fields.('sessionName')), 'UniformOutput', 0);
        
        for t = 1:length(nodeTypes)
            idx = cellfun(@(x) ~isempty(find(strcmp(nodeTypes{t}, x), 1)), events_pid(:, fields.('sessionName')), 'UniformOutput', 1);
            events_pid(idx, fields.('sessionName')) = repmat(nodeTypes(t), length(find(idx)), 1);
        end
 
        
        toc
        
        tic;
        fprintf(1, 'precompute summarize\n');
        % precompute summary by pid
        [summary_pid ops_pid] = summarizeByRow(events_pid, fields, allEventTypes);
        [TPIntervals, interval, ~] = resampleData(events_pid, fields, timeInterval, allEventTypes, 'eventType', timeRange);

	save(datafile, 'summary_pid', 'ops_pid', 'TPIntervals', 'interval', 'allEventTypes', 'allTypeNames', 'events_pid', 'fields', 'nodeTypes');       
 	toc
end

tic

        fprintf(1, 'summarize all procs\n');
        % aggregate by events then summarize
        [summary1, ops1] = summarizeAggregateRows(summary_pid, ops_pid);
        % summarize across rows
        [summary2, ops2] = summarizeAcrossRowsGaussian(summary_pid, ops_pid);
        [TP, ops_tp] = summarizeThroughput(TPIntervals, interval, allEventTypes);
        
        % and print it
        printSummary(cat(2, summary1, summary2, TP), cat(2, ops1, ops2, ops_tp), fid, allTypeNames, 'all', 'all');
       
      
        fprintf(1, 'summarize processes by type\n');
        for i = 1:length(nodeTypes)
            t = nodeTypes{i};

            %filter and identify row ids
            idx = strcmp(t, cat(1, events_pid(:, fields.('sessionName'))));
            [summary1, ops1] = summarizeAggregateRows(summary_pid(idx, :, :), ops_pid);
            %summarize across procs
            [summary2, ops2] = summarizeAcrossRowsGaussian(summary_pid(idx, :, :), ops_pid);

            [TP, ops_tp] = summarizeThroughput(TPIntervals(idx), interval, allEventTypes);

            printSummary(cat(2, summary1, summary2, TP), cat(2, ops1, ops2, ops_tp), fid, allTypeNames, 'nodeType', t);
            
        end
%         fprintf(1, 'summarize processes by hostname\n');
%         hostnames = unique(cat(1, events_1{:, fields.('hostName')}));
%         for i = 1:length(hostnames)
%             t = hostnames{i};
%
%             %filter and identify row ids
%             idx = find(strcmp(t, cat(1, events_pid{:, fields.('hostName')})));
%             %summarize across procs
%             [summary1, ops1] = summarizeAggregateRows(summary_pid(idx, :, :), ops_pid);
%             [summary2, ops2] = summarizeAcrossRowsGaussian(summary_pid(idx, :, :), ops_pid);
%             [TP, ops_tp] = summarizeThroughput(TPIntervals(idx), interval, allEventTypes);
%             printSummary(cat(2, summary1, summary2, TP), cat(2, ops1, ops2, ops_tp), fid, allTypeNames, 'procs', 'hostname', t);
%             
%         end
        

        toc
end










