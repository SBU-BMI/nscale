clear all;
close all;

% open the file
prefixes = {...
    '../strong-checkpoint-validation-ccicluster/strong-checkpoint-validation-ccicluster-timing',...
    '../weak-checkpoint-validation-ccicluster/weak-checkpoint-validation-ccicluster-timing.5',...
    '../weak-nocp-validation-ccicluster/weak-nocp-validation-ccicluster-timing.5',...
    };
% '../strong-nocp-validation-ccicluster/strong-nocp-validation-ccicluster-timing.5',...
%     '../strong-checkpoint-segtest-yellowstone/scio-segtest-mpi2-timing',...
%     '../strong-checkpoint-segtest-yellowstone/scio-segtest-mpi3-timing',...
%     '../strong-checkpoint-segtest-yellowstone/scio-segtest-mpi5-timing',...
%     '../strong-checkpoint-segtest-yellowstone/scio-segtest-mpi9-timing',...

nodeCountSuffixes = {'.5', '.9', '.17', '.33', '.65', '.129'};
nodeCounts = [4 8 16 32 64 128];

for i = 1:length(prefixes)
    
    % data aggregation
    workload = zeros(length(nodeCounts), 4);
    
    for k = 1:length(nodeCountSuffixes)
        prefix = [prefixes{i} nodeCountSuffixes{k}];
        
        [header type startt endt location] = readComputeAndIOTiming([prefix '.csv']);
        [ concurrencyAll loadAll nodes concurrencyPerNode loadPerNode procsPerNode ] = ...
            ComputeAndIOConcurrency( header, type, startt, endt, location );

        figure;
        title(prefix); hold on;
        xlabel('time'); hold on;
        ylabel('load'); hold on;
        plot(concurrencyAll{1,1}, concurrencyAll{1,2}(2, :), '-b'); hold on;
        plot(concurrencyAll{1,1}, concurrencyAll{1,2}(3, :), '-r'); hold on;
        legend('compute', 'io'); hold on;

        ofn = [prefix '.summary.csv'];
        dlmwrite(ofn, ['ALL'], 'delimiter', '');
        dlmwrite(ofn, concurrencyAll{1,1}, '-append', 'delimiter', ',', 'precision', '%ld');
        dlmwrite(ofn, concurrencyAll{1,2}, '-append', 'delimiter', ',');

        for j = 1 : length(nodes)
            dlmwrite(ofn, nodes{j}, '-append', 'delimiter', '');
            dlmwrite(ofn, concurrencyPerNode{j,1}{1,1}, '-append', 'delimiter', ',', 'precision', '%ld');
            dlmwrite(ofn, concurrencyPerNode{j,1}{1,2}, '-append', 'delimiter', ',');
        end

        save([prefix '.summary.mat'], 'prefix', 'concurrencyAll',...
            'concurrencyPerNode', 'location', 'nodes', 'loadAll',...
            'loadPerNode', 'procsPerNode');
        
        workload(k, :) = loadAll';
    end
    
    
    figure;
    title(prefixes{i}); hold on;
    xlabel('node counts'); hold on;
    ylabel('precent load'); hold on;
    plot(nodeCounts, workload(:, 3), '-b'); hold on;
    plot(nodeCounts, workload(:, 4), '-r'); hold on;
    legend('compute', 'io'); hold on;
    
    runname= prefixes{i};
    save([prefixes{i}, '.loads.mat'], 'runname', 'nodeCounts', 'workload',...
        'location', 'nodes', 'nodeCountSuffixes');
end