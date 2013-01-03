function summarizeFromIntermediate(summary_pid, ops_pid, TPIntervals, interval, allEventTypes, allTypeNames, fid)

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
%         fprintf(1, 'summarize processes by event name\n');
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



end










