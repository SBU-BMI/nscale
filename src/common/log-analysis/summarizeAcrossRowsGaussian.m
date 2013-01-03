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


    ops2 = {'min count', 'min total(ms)', 'min data(MB)', ...
        'max count', 'max total(ms)', 'max data(MB)', ...
        'mean count', 'mean total(ms)', 'mean data(MB)', ...
        'std count', 'std total(ms)', 'std data(MB)'};
 
    %% iterate over each row
    summary2 = zeros(size(summary, 2), length(ops2));   % number of rows is event types
        
    for i = 1:size(summary, 2)
        
        summary2(i, 1) = min(summary(:, i, 1));
        summary2(i, 2) = min(summary(:, i, 4));
        summary2(i, 3) = min(summary(:, i, 6));
        summary2(i, 4) = max(summary(:, i, 1));
        summary2(i, 5) = max(summary(:, i, 4));
        summary2(i, 6) = max(summary(:, i, 6));
        summary2(i, 7) = mean(summary(:, i, 1));
        summary2(i, 8) = mean(summary(:, i, 4));
        summary2(i, 9) = mean(summary(:, i, 6));
        summary2(i, 10) = std(summary(:, i, 1));
        summary2(i, 11) = std(summary(:, i, 4));
        summary2(i, 12) = std(summary(:, i, 6));

        
%         means = zeros(size(summary, 1),1);
%         stdev = zeros(size(summary, 1),1);
%         
%         idx = summary(:, i, 1) == 1;
%         if (~isempty(find(idx, 1)))
%             means(idx) = summary(idx, i, 4);
%             stdev(idx) = 0;
%         end
%         idx = summary(:, i, 1) > 1;
%         if (~isempty(find(idx, 1)))
%             means(idx) = summary(idx, i, 4) ./ summary(idx, i, 1);
%             stdev(idx) = summary(idx, i, 5) - (means(idx) .* means(idx) .* summary(idx, i, 1));
%             stdev(idx) = sqrt(stdev(idx) ./ (summary(idx, i, 1) - 1));
%         end
       
        
    end
    
    
end

