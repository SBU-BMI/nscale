close all;
clear all;

allEventTypes = [-1 0 11 12 21 22 23 31 32 41 42 43 44 45 46]';

allTypeNames = {'Other',...
    'Compute',...
    'Mem IO',...
    'GPU mem IO', ...
    'Network IO', ...
    'Network wait', ...
    'Network IO NB', ...
    'File read', ...
    'File write', ...
    'ADIOS init', ...
    'ADIOS open', ...
    'ADIOS alloc', ...
    'ADIOS write', ...
    'ADIOS close', ...
    'ADIOS finalize'};
        
colorMap = [0, 0, 0; ...              % OTHER, -1, black
            120.0, 0.4, 1.0; ...      % COMPUTE, 0, green
            240.0, 0.4, 1.0; ...      % MEM_IO, 11, blue
            240.0, 0.4, 1.0; ...      % GPU_MEM_IO, 12, blue
            180.0, 0.4, 1.0; ...      % NETWORK_IO, 21, cyan
            300.0, 0.4, 1.0; ...      % NETWORK_WAIT, 22, magenta
            180.0, 0.4, 1.0; ...      % NETWORK_IO_NB, 23, cyan
            60.0, 0.4, 1.0; ...       % FILE_I, 31, yellow
            0.0, 0.4, 1.0; ...        % FILE_O, 32, red
            180.0, 1.0, 1.0; ...      % ADIOS_INIT, 41, cyan
            300.0, 1.0, 1.0; ...      % ADIOS_OPEN, 42, magenta
            240.0, 1.0, 1.0; ...      % ADIOS_ALLOC, 43, blue
            60.0, 1.0, 1.0; ...       % ADIOS_WRITE, 44, yellow
            0.0, 1.0, 1.0; ...        % ADIOS_CLOSE, 45, red
            120.0, 1.0, 1.0; ...      % ADIOS_FINALIZE, 46, green
];
colorMap(:, 1) = colorMap(:, 1) / 180.0 * pi;  % in radian

dirs = {...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest.reverserandom.run',...
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param2', ... 
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param1', ... 
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params3', ...    
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-iprobefix.nb', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-iprobefix', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest.run2',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest.run1',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-randtime',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3-throughput',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-compressed-randtime', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-5', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-4', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-1', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-hpdc2012-weak1', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports-sep_osts', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong2', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-async1' ...
};

old_dirs = {... 
    '/home/tcpan/PhD/path/Data/adios/cci-gpuclus-blocking', ...
    '/home/tcpan/PhD/path/Data/adios/cci-gpuclus-b-compressed', ...
    '/home/tcpan/PhD/path/Data/adios/cci-gpu-clus-async-noADIOS', ...
    '/home/tcpan/PhD/path/Data/adios/cci-gpu-clus-async', ... 
    '/home/tcpan/PhD/path/Data/adios/cci-old-clus', ...
    '/home/tcpan/PhD/path/Data/adios/Jaguar-tcga4-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/Jaguar-tcga3-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/Jaguar-tcga2-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/Jaguar-tcga1-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small11-grouped-bench', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small10-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small9-grouped-debug', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small8-grouped', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small6-TP-barrier', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small7-TP-gapped-barrier', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small4-TP', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small5-TP-gapped-nobarrier', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small3-throughput',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga1',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga4-throughput-smallAMR', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params-debug', ... 
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-debug2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-debug', ...
    '/home/tcpan/PhD/path/Data/adios/yellowstone-B-compressed', ...
    '/home/tcpan/PhD/path/Data/adios/yellowstone-NB-buf4', ...
    '/home/tcpan/PhD/path/Data/adios/yellowstone-async', ...
    '/home/tcpan/PhD/path/Data/adios/yellowstone' ...
    };
    
%selections = [3 4 5];
selections = 1:length(dirs);

% canParallelize = 0;
% if iskeyword('matlabpool')
% 	canParallelize = 1;
% else
% 	fh = str2func('matlabpool');
% 	f = functions(fh);
% 	if (~isempty(f.file))
% 		canParallelize = 1;
% 	else
% 		canParallelize = 0;
% 	end
% end
% 
% if (canParallelize == 1)
%     if (matlabpool('size') <= 0)
%         matlabpool;
%     end
%     
%     fprintf(2, 'MATLAB parallel toolbox available.  using parfor loop\n');
%     
%     parfor j = 1 : length(selections)
%        %close all;
%        id = selections(j);
%        dirname = dirs{id};
% 
%        analyzeDir(dirname, allEventTypes, allTypeNames, colorMap);
% 
%     end
% 
%     
%     matlabpool close;
% else
% 
%     fprintf(2, 'MATLAB parallel toolbox not available.  using standard for loop\n');

            procFilter = cell(1,2);
            procFilter{1,1} = '~sessionName';
            procFilter{1,2} = {'m'};
            eventFilter = cell(1,2);
            eventFilter{1,1} = '~eventName';
            eventFilter{1,2} = {'MPI NB SEND', 'MPI NB RECV', 'MPI B SEND', 'MPI B RECV'}; 
            procAggreg = {'pid'};


        % each process right now can only be one type.
%         function z = myStringIntersect(x)
%         z = [];
%         for i = 1:length(nodeTypes)
%             if (strcmp(nodeTypes{i}, x))
%                 z = nodeTypes{i};
%             end
%         end
%         end
    
    for j = 1 : length(selections)
        id = selections(j);
        dirname = dirs{id};
    
        %analyzeDir(dirname, allEventTypes, allTypeNames, colorMap);
        

        % for testing
        files = dir(fullfile(dirname, '*.csv'));
        i = 1;
        [~, n, ~] = fileparts(files(i).name);
        prefix = fullfile(dirname, n);

        % old version is about 7% slower than the new version below.
%         fprintf(1, 'OLD VERSION\n');
%         proc_events = readComputeAndIOTimingOneLineFormat(dirname, files(i), procFilter{1,2}, eventFilter{1,2});
%         fprintf(1, 'OLD VERSION DONE\n');

        % read the data in
        tic
            fprintf(1, 'parsing %s/%s\n', dirname, files(i).name);
            [events fields] = readLog(dirname, files(i));
		size(events)
        toc

        if (size(events, 1) == 0)
            continue;
        end
        
%         tic
%         fprintf(1, 'filtering by process\n');
%         procFilter = cell(3,2);
%         procFilter{1,1} = '~sessionName';
%         procFilter{1,2} = {'pull', 'push'};
%         procFilter{2,1} = 'pid';
%         procFilter{2,2} = 0:99;
%         procFilter{3,1} = 'test';
%         procFilter{3,2} = [2, 5,6];
%         events2 = filterProcesses(events, fields, procFilter, 'and');
%     	size(events2)
%         toc
%         
%         tic;
%         fprintf(1, 'selection proces\n');
%         events3 = selectFields(events2, fields, {'pid', 'sessionName', 'startT'});
%     	size(events3)
%         toc
% 
%         tic
%         fprintf(1, 'aggregating by process\n');
%         eventAggreg = {'pid', 'hostName', 'sessionName', 'group'};
%         events4 = aggregateRows(events2, fields, eventAggreg);
%         toc
            
 
                    tic
            fprintf(1, 'filtering by process\n');
            events2 = filterProcesses(events, fields, procFilter, 'or');
            toc
            
            tic
            fprintf(1, 'filtering by event\n');
            events3 = filterEvents(events2, fields, eventFilter, 'or');
            toc
            
            tic
            fprintf(1, 'aggregating by process\n');
            events4 = aggregateRows(events3, fields, procAggreg);
            toc

            
fid = 1;            
proc_events = events;
                   
% get the min and max timestamps for the total time.
        % aggregate by pid
        tic;
        fprintf(1, 'loading data\n');
        [events_pid nodeTypes] = aggregatePerProc(proc_events, fields);
        toc
        

        %% compute intermediate results
        % compute min and max
        tic;
        fprintf(1, 'compute min and max\n');
        nrows = size(events_pid,1);
        mxx = ones(nrows, 1) * -1.0;
        mnx = ones(nrows, 1) * -1.0;
        et_id = fields.('endT');
        st_id = fields.('startT');
        for p = 1:nrows 
            if (~isempty(events_pid{p, et_id}))
                mxx(p) = max(events_pid{p, et_id}, [], 1);
            end
            if (~isempty(events_pid{p, st_id}))
                mnx(p) = min(events_pid{p, st_id}, [], 1);
            end
        end
        mx = max(mxx(mxx>=0), [], 1);  % maximum end timestamp
        mn = min(mnx(mnx>=0), [], 1)-1;  % min end timestamp
        durs = mxx - mnx;
        toc
          
        % intermediate summary by procs
        tic;
            fprintf(1, 'compute intermediate summary for events data based on pid.\n');
            [summary_pid ops_pid] = summarizeByRow(events_pid, fields, allEventTypes);
        toc
        
        
        % resample datasizes by events
        tic;
            fprintf(1, 'resampling data by events\n');
            [TPIntervals, interval, ~] = resampleData(events_pid, fields, timeInterval, allEventTypes, 'eventType', [mn mx]);
        toc

        

        
        
%% SUMMARIZE
        tic;
        fprintf(1, 'summarize\n');
        summarizeFromIntermeidate(summary_pid, ops_pid, TPIntervals, interval, allEventTypes, allTypeNames, fid);
        toc
    end

% end










