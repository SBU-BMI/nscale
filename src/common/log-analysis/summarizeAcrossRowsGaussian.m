function [summary2 ops2] = summarizeAcrossRowsGaussian( summary, ops)
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


    ops2 = {'mean count', 'mean min(ms)', 'mean max(ms)', 'mean total(ms)', 'mean mean(ms)', 'mean stdev(ms)',...
        'std count', 'std min(ms)', 'std max(ms)', 'std total(ms)', 'std mean(ms)', 'std stdev(ms)'};
 
    %% iterate over each row
%    attributes = cellfun(@(x) double(x)/(1024.0*1024.0), events(:, names.('attribute')), 'UniformOutput', 0);
    summary2 = zeros(size(summary, 2), length(ops2));   % number of rows is event types
        
    for i = 1:size(summary, 2)
%        attrs = cellfun(@(x,y) double(x(y)), attributes, idx, 'UniformOutput', 0);
        
        summary2(i, 1) = mean(summary(:, i, 1));
        summary2(i, 2) = mean(summary(:, i, 2));
        summary2(i, 3) = mean(summary(:, i, 3));
        summary2(i, 4) = mean(summary(:, i, 4));
    
        means = zeros(size(summary, 1),1);
        stdev = zeros(size(summary, 1),1);
        
        idx = summary(:, i, 1) == 1;
        if (~isempty(find(idx, 1)))
            means(idx) = summary(idx, i, 4);
            stdev(idx) = 0;
        end
        idx = summary(:, i, 1) > 1;
        if (~isempty(find(idx, 1)))
            means(idx) = summary(idx, i, 4) ./ summary(idx, i, 1);
            stdev(idx) = summary(idx, i, 5) - (means(idx) .* means(idx) .* summary(idx, i, 1));
            stdev(idx) = sqrt(stdev(idx) ./ (summary(idx, i, 1) - 1));
        end
       
        summary2(i, 5) = mean(means);
        summary2(i, 6) = mean(stdev);
        
        summary2(i, 7) = std(summary(:, i, 1));
        summary2(i, 8) = std(summary(:, i, 2));
        summary2(i, 9) = std(summary(:, i, 3));
        summary2(i, 10) = std(summary(:, i, 4));
        summary2(i, 11) = std(means);
        summary2(i, 12) = std(stdev);
        
    end
    
    
end

