filename = '/home/tcpan/PhD/path/Data/keeneland/adios-mpio-val-test.csv'
figname = '/home/tcpan/PhD/path/Data/keeneland/adios-mpio-val-test'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, figname);

pack;

figname = '/home/tcpan/PhD/path/Data/keeneland/posix-val-test'
filename = '/home/tcpan/PhD/path/Data/keeneland/posix-val-test.csv'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 50, 10000, figname);

pack;


filename = '/home/tcpan/PhD/path/Data/posix-seg-tests.csv'
figname = '/home/tcpan/PhD/path/Data/posix-seg-tests'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, figname);

pack;


filename = '/home/tcpan/PhD/path/Data/adios-seg-tests-NULL.csv'
figname = '/home/tcpan/PhD/path/Data/adios-seg-tests-NULL'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, figname);

pack;

filename = '/home/tcpan/PhD/path/Data/adios-seg-tests-POSIX.csv'
figname = '/home/tcpan/PhD/path/Data/adios-seg-tests-POSIX'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, figname);

pack;

filename = '/home/tcpan/PhD/path/Data/adios-seg-tests-MPI.csv'
figname = '/home/tcpan/PhD/path/Data/adios-seg-tests-MPI'
proc_events = readComputeAndIOTimingOneLineFormat(filename);
[img norm_events] = plotProcEvents(proc_events, 300, 1000, figname);

pack;