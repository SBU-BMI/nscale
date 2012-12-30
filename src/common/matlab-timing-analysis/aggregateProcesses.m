function [ out_events ] = aggregateProcesses( events, names, procid_fields )
% aggregateProcesses uses the procid_fields as unique identifiers and
% aggregate data base on unique values.
% if procid_fields is specified as '*', then aggregate all the rows
% together.

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
    allowed = intersect({'pid', 'hostName', 'group', 'sessionName'}, fns');

    
    if (~isempty(find(strcmp('*', procid_fields), 1)))
        clear out_events;
        for i = 1:size(events, 2)
            out_events{1, i} = cat(1, events{:, i});
        end
        numids = 0;
        fil = {};
        C = 1;
        IC = ones(size(events, 1), 1);
    else 
        fil = intersect(allowed, procid_fields);
        numids = length(fil);
        if (numids == 0)
            fprintf(1, 'no aggregation fields specified\n');
            return;
        end

        % get the allowed only
        ICs = zeros(size(events,1), numids, 'uint32');

        % get the individual column's uniqueness.  have to do this because cell
        % version of unique does not support rows keyword.
        for i = 1:numids
            name = fil{i};
            idx = names.(name);
            % need to use braces else we get a bunch of cells.
            values = cat(1, events{:, idx});  % need to cat else we only get 1 value.
            [~, ~, ICs(:, i)] = unique(values);  % get the unique value indices
        end

        % now do unique on the ICs matrix with "rows"
        [C, ~, IC] = unique(ICs, 'rows');  % get the unique index combinations
        % recall that ICs matches rows of events one to one.  it's another
        % encoding of the strings in events, essentially
        % so now C = ICs(IA), and ICs = C(IC)

        % create the output structure
        out_events = cell(size(C, 1), size(events, 2));
    end
    
    % identify the columns that need to have its rows merged.
    idxes = zeros(1, size(events, 2));
    for j = 1:length(allowed)
        idxes(names.(allowed{j})) = -1;
    end
    for j = 1:numids
        idxes(names.(fil{j})) = 1;
    end
    
    % merge the rows
    IC = uint32(IC);
    for j = 1:size(C, 1)
        pe2 = events(IC==j, :);
        % and only operate the columns that need merging.
        for i = find(idxes == 0)
            out_events{j, i} = cat(1, pe2{:, i});
        end
        
        for i = find(idxes == 1)
            out_events{j, i} = pe2{1, i};
        end
        
        for i = find(idxes == -1)
    %       out_events{j, i} = unique(cat(1, pe2{:, i}));
            temp = cat(1, pe2{:, i});
            temp = sort(temp);
            if (iscell(temp))
                if (ischar(temp{1}))
                    iidx = [true(1,1); ~strcmp(temp(1:end-1, 1), temp(2:end, 1))];
                elseif (isnumeric(temp{1}))
                    iidx = [true(1,1); temp(1:end-1,1) ~= temp(2:end, 1)];
                else
                    fprintf(2, 'ERROR: unknow data type\n');
                    continue;
                end
            else
                if (ischar(temp(1)))
                % string array
                    iidx = [true(1,1); ~strcmp(temp(1:end-1, 1), temp(2:end, 1))];                
                elseif (isnumeric(temp(1)))
                % NUMERIC ARRAY
                    iidx = [true(1,1); temp(1:end-1,1) ~= temp(2:end, 1)];
                else
                    fprintf(2, 'ERROR: unknow data type\n');
                    continue;
                end
            end
            out_events{j, i} = temp(iidx);
        end
    end
    
end

 
