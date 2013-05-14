%test2 - testing bar plot break point

close all;

figure;
BarPlotBreak(3, 1000, 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');



figure;
BarPlotBreak(gca, [1 2 3], [10,40,1000], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak(gca, [10,40,1000], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak(gca, [1 2 3], [10,40,1000], 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak(gca, [10,40,1000], 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak([1 2 3], [10,40,1000], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak([10,40,1000], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak([1 2 3], [10,40,1000], 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak([10,40,1000], 'break_range', [50, 60], 'break_type', 'RPatch');


figure;
BarPlotBreak([2 3], [10,40,1000; 20 60 300], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch', 'stacked');

figure;
BarPlotBreak([10,40,1000; 20 60 300], 0.5, 'break_range', [50, 60], 'break_type', 'RPatch', 'stacked');

figure;
BarPlotBreak([2 3], [10,40,1000; 20 60 300], 'break_range', [50, 60], 'break_type', 'RPatch', 'stacked');

figure;
BarPlotBreak([10,40,1000; 20 60 300], 'break_range', [50, 60], 'break_type', 'RPatch', 'stacked');


figure;
BarPlotBreak([2 3], [10,40,1000; 20 60 300], 0.5);

figure;
BarPlotBreak([10,40,1000; 20 60 300], 0.5);

figure;
BarPlotBreak([2 3], [10,40,1000; 20 60 300]);

figure;
BarPlotBreak([10,40,1000; 20 60 300]);


figure;
BarPlotBreak(3, 1000, 'break_range', [50, 60], 'break_type', 'RPatch');

figure;
BarPlotBreak(1000, 0.5, 'break_range', [50, 60], 'break_type', 'RPatch');

