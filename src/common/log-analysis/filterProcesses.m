function [ out_events ] = filterProcesses( events, names, filter, joinType )
% selects process based on process identifiers or characteristics.  
% this is a simple placeholder for  better database style WHERE clause
% filter is an nx2 cell array with first column containing fieldnames, and
% second column containing values.
% joinTypes are AND or OR to join the different fields
% For each field, a list of selected values (in cell array) are used -
% these are ORed implicitly.
% support equality check only, no range query.
% support negation but not simultaneously negate and not negate on the same
% field (does not make sense.  the field values are not multivalued)
% extracts specifies which fields to extract. {'*'} and {} mean all fields.

    
    % set up the output structure
    out_events = events;

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
    allowed = intersect({'pid', 'hostName', 'group', 'sessionName'}, fns');


    % check to make sure there are things to filter by
    if (size(filter, 1) == 0)
        fprintf(1, 'no filter specified\n');
        return;
    end

    % separately mark the negations
    fil = filter;
    marked = zeros(size(fil, 1), 'int8');
    for i = 1:size(fil,1)
        fn = fil{i, 1};
        si = 1;
        if (strncmp(fn, '~', 1))
            fn = fn(2:end);
            si = -1;
        end
        if (~isempty(find(strcmp(fn, allowed'), 1)))
            if (marked(i) == 0)
                marked(i) = si;
            else
                fprintf(2, '%s and ~%s are both specified as a filter. UNSUPPORTED\n', fn, fn);
                return;
            end
            fil{i, 1} = fn;
        else
            fprintf(2, '%s is not allowed as an identifier.\n', fn);
            continue;
        end	
    end
    fil = fil(marked~=0, :);
    marked = marked(marked~=0);

    % check to make sure join type is correct
    if (strcmpi('AND', joinType) == 1)
        jt = 1;
        selected = true(size(events, 1), size(fil, 1));
    elseif (strcmpi('OR', joinType) == 1)
        jt = 0;
        selected = false(size(events, 1), size(fil, 1));
    else
        fprintf(2, 'unsupported joinType specified: %s\n', joinType);
        return;
    end
    
    
    
    % check to see if we have filters that need to be ignored.
    % negation is tricky - within the field the positives should be ORed
    % then negated, else do it one by one. when joining between fields,
    % need to have separate data structure for AND.
    for i = 1:size(fil, 1)
        name = fil{i, 1};

        idx = names.(name);
       
        % need to use braces else we get a bunch of cells.
        values = cat(1, events{:, idx});  % need to cat else we only get 1 value.
        s = ismember(values, fil{i, 2}');  % implicit OR of the values specified,
        % implicit AND of the negative values specified
    
        if (marked(i) == -1)
            s = ~s;
        end
        
        selected(:, i) = s;
    end

    
    rows = true(size(selected, 1), 1);
    if (jt == 1)
        rows = all(selected, 2);
    elseif (jt == 0)
        rows = any(selected, 2);
    end


    out_events(~rows, :) = [];
    clear rows;
    clear s;
    clear idx;
    clear name;
    clear values;
    clear jt;
    clear selected;
end

 
