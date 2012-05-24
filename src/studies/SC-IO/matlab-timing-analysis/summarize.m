function [ ] = summarize( proc_events )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    p = size(proc_events, 1);

    mn = inf;
    mx = 0;
    for i = 1:p 
	   mn = min([mn, min(proc_events{i, 6})]);
       mx = max([mx, max(proc_events{i, 7})]);
    end

    
    % init variables
    nTiles = zeros(p, 1);
    readTotals = zeros(p, 1);
    readDurations = [];
    computeTotals = zeros(p, 1);
    computeDurations = [];
    barrierTotals = zeros(p, 1);
    barrierDurations = [];

    writeTotals = zeros(p, 1);
    writeDurations = [];
    
    % per node info
    for i = 1:p-1
        names = proc_events{i, 4};
        types = proc_events{i, 5};
        startt = proc_events{i, 6};
        endt = proc_events{i, 7};

        % tile total
        nTiles(i) = sum(types == 0, 1);
        
        % compute total
        idx = find(types == 0);
        duration = endt(idx) - startt(idx);
        computeTotals(i) = sum(duration,1);
        computeDurations = [computeDurations; duration];
        
        % read total
        idx = find(types == 31);
        duration = endt(idx) - startt(idx);
        readTotals(i) = sum(duration,1);
        readDurations = [readDurations; duration];

        
        % barrier total
        idx = find(strcmp(names, 'IO MPI scan') | strcmp(names, 'IO MPI allreduce'));
        duration = endt(idx) - startt(idx);
        barrierTotals(i) = sum(duration,1);
        barrierDurations = [barrierDurations; duration];
        
    
        % write total
        idx = find(types == 42 | types == 44 | types == 45 | types == 32 );
        duration = endt(idx) - startt(idx);
        writeTotals(i) = sum(duration,1);
        writeDurations = [writeDurations; duration];

    
    end
    
    readTotal = sum(readTotals, 1)
    readPercent = readTotal / mx
    readMean = mean(readDurations)
    readStdev = std(double(readDurations))
    
    computeTotal = sum(computeTotals, 1)
    computePercent = computeTotal / mx
    computeMean = mean(computeDurations)
    computeStdev = std(double(computeDurations))
    
    barrierTotal = sum(barrierTotals, 1)
    barrierPercent = barrierTotal / mx
    barrierMean = mean(barrierDurations)
    barrierStdev = std(double(barrierDurations))
    
    
    writeTotal = sum(writeTotals, 1)
    writePercent = writeTotal / mx
    writeMean = mean(writeDurations)
    writeStdev = std(double(writeDurations))

    
end

