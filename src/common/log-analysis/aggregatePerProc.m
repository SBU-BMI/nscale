function [events_pid nodeTypes ] = aggregatePerProc(proc_events, fields)

    tic
    fprintf(1, 'aggregate events into pid based and relabel sessions\n');
    % group by pid first
    events_pid = aggregateRows(proc_events, fields, {'pid'});
    % now replace the sessionNames.
    sessions = unique(cat(1, proc_events{:, fields.('sessionName')}));
    % next look for the right set
    if (length(intersect(sessions, {'m', 'w'})) == 2)
        nodeTypes = {'m', 'w'};
    elseif (length(intersect(sessions, {'read', 'io', 'seg'})) == 3)
        nodeTypes = {'read', 'io', 'seg'};
    elseif (length(intersect(sessions, {'assign', 'io', 'seg'})) == 3)
        nodeTypes = {'assign', 'io', 'seg'};
    elseif (length(intersect(sessions, {'io', 'seg'})) == 2)
        nodeTypes = {'io', 'seg'};
    else
        fprintf(2, 'ERROR: log does not have appropriate node session types\n  ');
        for s = 1:length(sessions)
            fprintf(2, '%s,', sessions{s});
        end
        fprintf(2, '\n');
        return;
    end
    
    % THIS CODE IS STUPID SLOW.  USE ALTERNATIVE BELOW.
    %       events_pid(:, fields.('sessionName')) = ...
    %           cellfun(@(x) intersect(nodeTypes, x), events_pid(:, fields.('sessionName')), 'UniformOutput', 0);
    
    for t = 1:length(nodeTypes)
        idx = cellfun(@(x) ~isempty(find(strcmp(nodeTypes{t}, x), 1)), events_pid(:, fields.('sessionName')), 'UniformOutput', 1);
        events_pid(idx, fields.('sessionName')) = repmat(nodeTypes(t), length(find(idx)), 1);
    end
    toc

end










