function [proc_summary, ops] = summarizeByRow( events, names, allEventTypes)
%summarizeByRow summarizes the log data by event type
% for now, just do the simple min, max, mode, median, mean, stdev, and total.  each row is
% summarized separately, across row summary is done in post processing
% (excel)
%
% if separate entries are needed, (e.g. for each host) then use filterProcess first.
%
% 'all' in aggregateField means there should be only 1 entry and the output
% should use the field value of 'all'.

    %% check the parameters.
    if (size(events, 1) == 0)
        fprintf(2, 'no entries in input event log list\n');
        return;
    end
    
    if (size(events, 2) == 0)
        fprintf(2, 'no field in input event log list\n');
        return;
    end

    % check to make sure there are things to filter by
    fns = fieldnames(names);
    target = intersect({'eventType', 'startT', 'endT', 'attribute'}, fns');
    
    
    ops = {'count', 'min(ms)', 'max(ms)', 'total(ms)', 'sumsquares', 'data(mb)'};

    
 
    %% iterate over each row
    durations = cellfun(@(x,y) double(x-y)/1000.0, events(:, names.('endT')), events(:, names.('startT')), 'UniformOutput', 0);
    event_types = events(:, names.('eventType'));
    attributes = cellfun(@(x) double(x)/(1024.0*1024.0), events(:, names.('attribute')), 'UniformOutput', 0);
    proc_summary = zeros(size(events, 1), length(allEventTypes), length(ops));
        
    
    for i = 1:length(allEventTypes)
        t = allEventTypes(i);
        for j = 1:size(durations, 1)
             idx = find(event_types{j} == t);
             if (isempty(idx))
                 continue;
             end
             durs = durations{j}(idx);
             attrs = attributes{j}(idx);
             
                proc_summary(j, i, 1) = length(durs);
                proc_summary(j, i, 2) = min(durs);
                proc_summary(j, i, 3) = max(durs);
                proc_summary(j, i, 4) = sum(durs);
                proc_summary(j, i, 5) = sum(durs .* durs);
                proc_summary(j, i, 6) = sum(attrs);
         end
    end
  
end

