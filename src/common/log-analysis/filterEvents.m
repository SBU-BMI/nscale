function [ out_events ] = filterEvents( events, names, filter, joinType )
% selects process based on event types or characteristics.  
% this is a simple placeholder for  better database style WHERE clause
% filter is an nx2 cell array with first column containing fieldnames, and
% second column containing values.
% joinTypes are AND or OR to join the different fields
% For each field, a list of selected values (in cell array) are used -
% these are ORed implicitly.
% support equality check only, no range query. 
% support negation but not simultaneously negate and not negate on the same
% field (does not make sense.  the field values are not multivalued)


    
    % check to make sure there are things to filter
    out_events = events;
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
    allowed = intersect({'eventName', 'eventType'}, fns');
    target = intersect({'eventName', 'eventType', 'startT', 'endT', 'attribute'}, fns');

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
    jt = -1;
    if (strcmpi('AND', joinType) == 1)
        jt = 1;
    elseif (strcmpi('OR', joinType) == 1)
        jt = 0;
    else
        fprintf(2, 'unsupported joinType specified: %s\n', joinType);
        return;
    end
    
    
    % negation is tricky - within the field the positives should be ORed
    % then negated, else do it one by one. when joining between fields,
    % need to have separate data structure for AND.

    % in place of a multiline anon function. x is a vector
    function [z] = testmember(x)
        if (isempty(y))
            z = true(size(x));
            return;
        end
        z = false(size(x));
        if (~isempty(x))
            if (iscell(y))
                for k = 1:length(y)
                    z = or(z, strcmp(y{k}, x));
                end
            else
                for k = 1:length(y)
                    z = or(z, (x == y(k)));
                end
            end
            
            % too slow: 85.5% of the time spent in
            %this function.
            %t = ismember(x, y);  
            
            % even slower below
%             [ux, ~, ic] = unique(x);
%             t = ismember(ux, y);
%             r = ismember(ic, find(t));

            if (marked(i) == -1)
                z = ~z;
            end
            return;
        end 
    end
    
    selected = {};
    for i = 1:size(fil, 1)
        % check to see if we have filters that need to be ignored.
        name = fil{i, 1};
        y = unique(fil{i,2});
        idx = names.(name);
        
        % need to use parens else we get a single array.
        values = cat(1, events(:, idx));  % need to cat else we only get 1 value.
        s = cellfun(@testmember, values, 'UniformOutput', 0); % implicit OR of the values specified

% not faster
%         t = cell(size(values, 1), 1);
%         for k = 1:size(values,1)
%             t{k} = testmember(values{k});
%         end
        
        if (isempty(selected))
            selected = s;
        else
            if (jt == 1)
                selected = cellfun(@and, selected, s, 'UniformOutput', 0);
            elseif (jt == 0)
                selected = cellfun(@or, selected, s, 'UniformOutput', 0);
            end
        end
    end
    
    function [z] = selectmember(x, y)
        %if (size(x, 1))
        if (~isempty(x) && ~isempty(y))
            z = x(y);
            return;
        else
            z = x;  % empty x, so return it.
            return;
        end 
    end

    selected = cellfun(@find, selected, 'UniformOutput', 0);    
    for i = 1:length(target)
        idx = names.(target{i});
        values = cat(1, events(:, idx));
        out_events(:, idx) = cellfun(@selectmember, values, selected, 'UniformOutput', 0);

% not faster
%             t = cell(size(values, 1), 1);
%             for k = 1:size(values,1)
%                 t{k} = selectmember(values{k}, selected{k});
%             end
        
    end
    
    clear rows;
    clear s;
    clear idx;
    clear name;
    clear values;
    clear jt;
    clear selected;
end

 
