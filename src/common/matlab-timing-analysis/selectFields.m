function [ out_events, out_names ] = selectFields( events, names, out_field_names)
% selects fields .  also generates a new set of nname to index mapping

    
    % check to make sure there are things to filter
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
    if (isempty(out_field_names) || ~isempty(find(strcmp('*', out_field_names), 1)))
        colnames = fns;
    else
        colnames = intersect(out_field_names, fns);    
    end

    
    
    % set up the output structure.  don't need to have same order as names.
    cols = zeros(length(colnames),1);
    for i = 1:length(colnames)
       cols(i) = names.(colnames{i});
       out_names.(colnames{i}) = i;
    end
    out_events = events(:, cols);

end
    
 
