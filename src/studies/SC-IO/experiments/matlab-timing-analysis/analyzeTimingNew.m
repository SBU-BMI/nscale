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
