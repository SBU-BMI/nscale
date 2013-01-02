function printSummary( summary, ops, fid, allTypeNames, aggregField, fieldVal)
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

    % if no fid specified, print out to console
    if (isempty(fid) || fid < 1)
        fid = 1;
    end
       
    % output now.
    fprintf(fid, 'field, value, event_type');
    for i = 1:length(ops)
        fprintf(fid, ', %s', ops{i});
    end
    fprintf(fid, '\n');
    
  
    for i = 1:size(summary, 1)

        if (iscell(fieldVal))
            fprintf(fid, '%s, %s, %s', aggregField, fieldVal{1}, allTypeNames{i});
        elseif (ischar(fieldVal))
            fprintf(fid, '%s, %s, %s', aggregField, fieldVal, allTypeNames{i});
        else
            fprintf(fid, '%s, %d, %s', aggregField, fieldVal, allTypeNames{i});
        end
        for j = 1:length(ops)
            fprintf(fid, ', %f', summary(i, j));
        end        
        fprintf(fid, '\n');
    end
    
    
end

