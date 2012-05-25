clear all;

timeInterval = 20000; % 20 millisec
procWidth = 20;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/posix-tcga_small'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/adios-tcga_small-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/adios-tcga_small-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/adios-tcga_small-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/adios-tcga_small-MPI_LUSTRE'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small1/adios-tcga_small-MPI_AMR'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;



timeInterval = 20000; % 20 millisec
procWidth = 20;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/posix-tcga_small'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/adios-tcga_small-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/adios-tcga_small-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/adios-tcga_small-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/adios-tcga_small-MPI_LUSTRE'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga-small2/adios-tcga_small-MPI_AMR'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;



timeInterval = 100000; % 100 millisec
procWidth = 20;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/posix-tcga'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/adios-tcga-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/adios-tcga-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

% prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/adios-tcga-MPI'
% proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
% [img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
% 
% prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/adios-tcga-MPI_LUSTRE'
% proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
% [img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
% 
% prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga/adios-tcga-MPI_AMR'
% proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
% [img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;

timeInterval = 100000; % 100 millisec
procWidth = 20;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/posix-tcga'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/adios-tcga-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/adios-tcga-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/adios-tcga-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/adios-tcga-MPI_LUSTRE'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga2/adios-tcga-MPI_AMR'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;



timeInterval = 100000; % 100 millisec
procWidth = 20;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/posix-tcga'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end
    
prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/adios-tcga-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/adios-tcga-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/adios-tcga-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/adios-tcga-MPI_LUSTRE'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland-tcga3/adios-tcga-MPI_AMR'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);
if size(proc_events, 2) == 8
    [sampled_events, sum_events, data_sizes, sum_data] = summarize(proc_events, timeInterval);
end

clear all;



timeInterval = 1000; % 1 millisec
procWidth = 300;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/posix-seg-tests'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, procWidth, timeInterval, prefix);

clear all;
