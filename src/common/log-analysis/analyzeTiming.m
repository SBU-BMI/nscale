close all;
clear all;

allEventTypes = [-1 11 12 41 46 42 43 0 21 22 23 31 32 44 45 ]';

allTypeNames = {'Other',...
    'Mem IO',...
    'GPU mem IO', ...
    'ADIOS init', ...
    'ADIOS finalize', ...
    'ADIOS open', ...
    'ADIOS alloc', ...
    'Compute',...
    'Network IO', ...
    'Network wait', ...
    'Network IO NB', ...
    'File read', ...
    'File write', ...
    'ADIOS write', ...
    'ADIOS close'};

colorMap = [0, 0, 0; ...              % OTHER, -1, black
            300.0, 0.5, 1.0; ...      % MEM_IO, 11, magenta
            300.0, 0.5, 1.0; ...      % GPU_MEM_IO, 12, magenta
            240.0, 0.5, 1.0; ...      % ADIOS_INIT, 41, blue
            240.0, 0.5, 1.0; ...      % ADIOS_FINALIZE, 46, blue
            180.0, 0.5, 1.0; ...      % ADIOS_OPEN, 42, cyan
            300.0, 1.0, 1.0; ...      % ADIOS_ALLOC, 43, magenta
            120.0, 1.0, 1.0; ...      % COMPUTE, 0, green
            180.0, 0.75, 1.0; ...      % NETWORK_IO, 21, cyan
            240.0, 0.75, 1.0; ...      % NETWORK_WAIT, 22, blue
            270.0, 0.75, 1.0; ...      % NETWORK_IO_NB, 23, purple
            60.0, 0.75, 1.0; ...       % FILE_I, 31, yellow
            0.0, 0.75, 1.0; ...        % FILE_O, 32, red
            30.0, 0.75, 1.0; ...       % ADIOS_WRITE, 44, orange
            0.0, 0.75, 1.0; ...        % ADIOS_CLOSE, 45, red
];
colorMap(:, 1) = colorMap(:, 1) / 360.0;  % in radian

lineTypes = {'--k', ...  
    '-.g', ...
    ':b', ...
    ':b', ...
    '-.c', ...
    ':m', ...
    '-.c', ...
    ':y', ...
    '-.r', ...
    '-c', ...
    '-m', ...
    '-b', ...
    '-y', ...
    '-r', ...
    '-g'...
    };

dirs = {...
    '/home/tcpan/PhD/path/Data/adios/tcga.titan.p10240.1', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.titan.p2048.3', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.titan.p2048.2', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.titan.p2048.1', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.1', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.2', ... 
    '/home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.3', ... 
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param2', ... 
    '/home/tcpan/PhD/path/Data/adios/kfs-syntest-param1', ... 
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params3', ...    
    '/home/tcpan/PhD/path/Data/adios/keeneland-syntest-params2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b3', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b2', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b' ...
};
pre_iprobe_dirs = {...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-compress-cap3', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-compress-cap2', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-compress-cap1', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-compressed1', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-5', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-4', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-3', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-2', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-hpdc2012-1', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-hpdc2012-weak1', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-transports-sep_osts', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-transports', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-strong2', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-strong1', ...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-syntest.reverserandom.run',...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-syntest.run2',...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-syntest.run1',...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-randtime',...
    '/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-compressed-randtime' ...
    };

old_dirs = {... 
    '/home/tcpan/PhD/path/Data/adios/jaguar-tcga-async1', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3-throughput',...
    '/home/tcpan/PhD/path/Data/adios/keeneland-iprobefix.nb', ...
    '/home/tcpan/PhD/path/Data/adios/keeneland-iprobefix', ...
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
          
errorfid = fopen('error.log', 'w');
%errorfid = 2; 
% selections = 1:length(dirs);
% selections = 1:2;
% for j = 1 : length(selections)
%         id = selections(j);
%         dirname = dirs{id};
%     
%         analyzeDir(dirname, allEventTypes, allTypeNames, errorfid);
%         %renderDir(dirname, allEventTypes, colorMap, lineTypes, errorfid);
%         %checkDir(dirname, allEventTypes, allTypeNames, colorMap, errorfid);
%     end

selections = 1:length(pre_iprobe_dirs);
selections = 2;
    for j = 1 : length(selections)
        id = selections(j);
        dirname = dirs{id};
    
        analyzeDir(dirname, allEventTypes, allTypeNames, errorfid);
        renderDir(dirname, allEventTypes, colorMap, lineTypes, errorfid);
        %checkDir(dirname, allEventTypes, allTypeNames, colorMap, errorfid);
    end
fclose(errorfid);
    
% end










