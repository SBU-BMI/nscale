function ExperimentVisualAnalysis( events, fields, figname_prefix, allEventTypes, colorMap)
% ExperimentVisualAnalysis: temporary - for visualizing correlations


%% constants


temp = colorMap;
temp(:, 3) = 0.95;
colorMapRGB2 = hsv2rgb(temp);


%addpath('/home/tcpan/PhD/path/src/nscale/src/common/log-analysis/allfitdist/');
%addpath('/home/tcpan/PhD/path/src/nscale/src/common/log-analysis/BreakPlot/')
%addpath('/home/tcpan/PhD/path/src/nscale/src/common/log-analysis/BreakBarPlot/')

%% load data
ionode_idx = strcmp('io', events(:, fields.('sessionName')));
io_event_types = cat(1, events{ionode_idx, fields.('eventType')});
io_event_names = cat(1, events{ionode_idx, fields.('eventName')});
io_event_starts = double(cat(1, events{ionode_idx, fields.('startT')})) / 1000000;
io_event_ends = double(cat(1, events{ionode_idx, fields.('endT')})) / 1000000;
io_event_durs = io_event_ends - io_event_starts + 1;

segnode_idx = strcmp('seg', events(:, fields.('sessionName')));
seg_event_types = cat(1, events{segnode_idx, fields.('eventType')});
seg_event_names = cat(1, events{segnode_idx, fields.('eventName')});
seg_event_starts = double(cat(1, events{segnode_idx, fields.('startT')})) / 1000000;
seg_event_ends = double(cat(1, events{segnode_idx, fields.('endT')})) / 1000000;
seg_event_durs = seg_event_ends - seg_event_starts;

fig = figure('visible', 'off');

%% plot compute
comp_only_idx = seg_event_types == 0;
comp_only_durs = seg_event_durs(comp_only_idx);
subplot(2, 3, 1);
[N, X] = hist(seg_event_starts(comp_only_idx), 1000);
mx = max(N);
med = median(N);
%BarPlotBreak(X, N, 2*med, mx - 2*med, 'Patch', );
bar(X, N);
title('Compute event start/end timestamp histogram');
% [D PD] = allfitdist(comp_only_durs, 'AICc', 'PDF');

compFull_only_idx = strcmp('computeFull', seg_event_names);
compFull_only_durs = seg_event_durs(compFull_only_idx);
compNoFG_only_idx = strcmp('computeNoFG', seg_event_names);
compNoFG_only_durs = seg_event_durs(compNoFG_only_idx);
compNoNU_only_idx = strcmp('computeNoNU', seg_event_names);
compNoNU_only_durs = seg_event_durs(compNoNU_only_idx);
% [D PD] = allfitdist(single(compFull_only_durs), 'AICc', 'PDF');
% [D PD] = allfitdist(single(compNoFG_only_durs), 'AICc', 'PDF');
% [D PD] = allfitdist(single(compNoNU_only_durs), 'AICc', 'PDF');

subplot(2, 3, 4);
[N0, X] = hist(comp_only_durs, 1000);
N1 = hist(compNoFG_only_durs, X);
N2 = hist(compNoNU_only_durs, X);
N3 = hist(compFull_only_durs, X);
mx = max(N);
med = median(N);
%breakplot(X, N, 2*med, mx - 2*med, 'Patch');

bar(X', cat(2, N1', N2', N3'), 'stacked');
title('types of compute stacked bar');
% hold on;
% plot(X, N0, 'k-');


%% plot read
read_only_idx = strcmp('read', seg_event_names);
read_only_durs = seg_event_durs(read_only_idx);
subplot(2, 3, 5); hist(read_only_durs, 1000); title('Read event time histogram');
subplot(2, 3, 2); hist(seg_event_starts(read_only_idx), 1000); title('read event start/end timestamp histogram');
%[D PD] = allfitdist(single(read_only_durs), 'AICc', 'PDF');


%% plot adios write out
adiosClose_only_idx = strcmp('adios close', io_event_names);
adios_only_durs = io_event_durs(adiosClose_only_idx);
subplot(2, 3, 6); hist(adios_only_durs, 1000); title('Adios event time histogram');
subplot(2, 3, 3); hist(io_event_starts(adiosClose_only_idx), 1000); title('adios event start/end timestamp histogram');
%[D PD] = allfitdist(adios_only_durs, 'PDF', 'AICc');

set(fig, 'InvertHardCopy', 'off');
set(fig, 'paperposition', [0 0 11 8.5]);
set(fig, 'papersize', [11 8.5]);
resolutionScaling = 3;
print(fig, '-dpng', [figname_prefix '.hist.png'], ['-r' num2str(100 * resolutionScaling)]);




fig = figure('visible','off');
%% plot network IO times
nio_only_idx = seg_event_types == 21;
nio_only_durs = seg_event_durs(nio_only_idx);
subplot(2, 2, 3); hist(nio_only_durs, 1000); title('NetIO event time histogram');
subplot(2, 2, 1); hist(seg_event_starts(nio_only_idx), 1000); title('NetIO event start/end timestamp histogram');
%[D PD] = allfitdist(adios_only_durs, 'PDF', 'AICc');


%% plot network IO times
nionb_only_idx = seg_event_types == 23;
nionb_only_durs = seg_event_durs(nionb_only_idx);
subplot(2, 2, 4); hist(nionb_only_durs, 1000); title('NetIONB event time histogram');
subplot(2, 2, 2); hist(seg_event_starts(nionb_only_idx), 1000); title('NetIONB event start/end timestamp histogram');
%[D PD] = allfitdist(adios_only_durs, 'PDF', 'AICc');



set(fig, 'InvertHardCopy', 'off');
set(fig, 'paperposition', [0 0 11 8.5]);
set(fig, 'papersize', [11 8.5]);
resolutionScaling = 3;
print(fig, '-dpng', [fignameprefix '.net_hist.png'], ['-r' num2str(100 * resolutionScaling)]);


%% plot start and end times.
maxtime = max([seg_event_ends; io_event_ends]);


fig = figure('visible', 'off');
% subplot(1,3, 1);
% plot([0 maxtime], [0, maxtime], 'k-');
% hold on; plot(seg_event_starts(comp_only_idx), seg_event_ends(comp_only_idx), 'g.');
% hold on; plot(io_event_starts(adiosClose_only_idx), io_event_ends(adiosClose_only_idx), 'r.');
% hold on; plot(seg_event_starts(read_only_idx), seg_event_ends(read_only_idx), 'b.');
% axis square;
% title('end vs start times for compute (g), write (r), and read (b)');


subplot(1,2,1);
hold on;
plot([0 maxtime], [0, maxtime], 'k-');
for c = 1:length(allEventTypes)
    idx = seg_event_types == allEventTypes(c);
    plot(seg_event_starts(idx), seg_event_ends(idx), '.', 'color', colorMapRGB2(c, :), 'MarkerSize', 2);hold on;
end
axis square;
title({'end vs start times for seg node events.', ... 
    'network IO (cyan), NB network IO(purple), compute(green), read(yellow)'});hold on;

subplot(1,2,2);
hold on;
plot([0 maxtime], [0, maxtime], 'k-');
for c = 1:length(allEventTypes)
    idx = io_event_types == allEventTypes(c);
    plot(io_event_starts(idx), io_event_ends(idx), '.', 'color', colorMapRGB2(c, :), 'MarkerSize', 2); hold on;
end
axis square;
title({'end vs start times io node events.',...
    'network IO (cyan), NB network IO(purple), write/adios close (red), adios write(orange)'}); hold on;


set(fig, 'InvertHardCopy', 'off');
set(fig, 'paperposition', [0 0 11 8.5]);
set(fig, 'papersize', [11 8.5]);
print(fig, '-dpng', [figname_prefix '.eventtimes.png'], ['-r' num2str(100 * resolutionScaling)]);

end