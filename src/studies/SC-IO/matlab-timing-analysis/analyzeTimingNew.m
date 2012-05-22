clear all;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/posix-tcga_small'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/adios-tcga_small-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/adios-tcga_small-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/adios-tcga_small-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/adios-tcga_small-MPI_LUSTRE'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/keeneland/adios-tcga_small-MPI_AMR'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, prefix);

pack;






prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/posix-seg-tests'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-NULL'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, prefix);

pack;

prefix = '/home/tcpan/PhD/path/Data/adios/yellowstone/adios-seg-tests-MPI'
proc_events = readComputeAndIOTimingOneLineFormat([prefix '.csv']);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, prefix);

pack;