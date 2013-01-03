function [summary2 ops2] = summarizeAggregateRows(summary, ops)
%summarizeLogs summarizes the log data by event type
% for now, just do the simple min, max, mode, median, mean, stdev, and total.  each row is
% summarized separately, across row summary is done in post processing
% (excel)
%
% if separate entries are needed, (e.g. for each host) then use filterProcess first.
%
% 'all' in aggregateField means there should be only 1 entry and the output
% should use the field value of 'all'.

    %% check the parameters.


    ops2 = {'count', 'min(ms)', 'max(ms)', 'total(ms)', 'mean(ms)', 'stdev(ms)', 'total data(MB)'};
 
    %% iterate over each row
%    attributes = cellfun(@(x) double(x)/(1024.0*1024.0), events(:, names.('attribute')), 'UniformOutput', 0);
    summary2 = zeros(size(summary, 2), length(ops2));   % number of rows is event types
        
    for i = 1:size(summary, 2)
%        attrs = cellfun(@(x,y) double(x(y)), attributes, idx, 'UniformOutput', 0);
        
        summary2(i, 1) = sum(summary(:, i, 1));
        summary2(i, 2) = min(summary(:, i, 2));
        summary2(i, 3) = max(summary(:, i, 3));
        summary2(i, 4) = sum(summary(:, i, 4));
        if (summary2(i, 1) == 0)
            summary2(i, 5) = 0;
            summary2(i, 6) = 0;
        elseif (summary2(i, 1) == 1)
            summary2(i, 5) = summary2(i, 4);
            summary2(i, 6) = 0;
        else
            summary2(i, 5) = summary2(i, 4) / summary2(i, 1);
            summary2(i, 6) = sum(summary(:, i, 5)) - (summary2(i, 5) * summary2(i, 5) * summary2(i, 1));
            summary2(i, 6) = sqrt(summary2(i, 6) / (summary2(i, 1) - 1));        
        end
        summary2(i, 7) = sum(summary(:, i, 6));
    end
    
    
end

