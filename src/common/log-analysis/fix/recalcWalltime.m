close all;
clear all;


dirs = {...
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param2', ...
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compress-cap1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-5', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-4', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3-throughput',...
    '/home/tcpan/PhD/path/Data/adios/jaguar-hpdc2012-weak1', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports-sep_osts', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong2', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1', ...
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-async1' ...
    };


%selections = [3 4 5];
selections = 1:length(dirs);


for j = 1 : length(selections)
    id = selections(j);
    dirname = dirs{id};
    
    summaryFilename = [dirname, '.summary.v2.walltimes.csv'];

    fid = fopen(summaryFilename, 'w');
    fclose(fid);

    
    % for testing
    files = dir(fullfile(dirname, '*.csv'));
    
    for i = 1:length(files)
        [~, n, ~] = fileparts(files(i).name);
        prefix = fullfile(dirname, n);
        fprintf(1, 'filename is %s\n', prefix);
        
        if (exist([prefix '.events_pid.mat']) == 2)
            load([prefix '.events_pid.mat']);
            
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
            
            fprintf(1, 'summarizing\n');
            fid = fopen(summaryFilename, 'a');
            fprintf(fid, 'EXPERIMENT, %s, app wall time, %f, sum process wall time, %f\n', prefix, mx-mn, sum(durs));
            
            
            fclose(fid);
            
            
            clear summary_pid ops_pid allEventTypes allTypeNames events_pid fields nodeTypes TPIntervals interval;
        else
            fprintf(1, 'NO MATCHING %s\n', prefix);
        end
        
    end
end

% end










