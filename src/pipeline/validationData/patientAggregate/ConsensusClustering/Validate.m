%compares HPC pipeline results with matlab results for nuclear segmentation
%of TCGA GBM permanent sections.

close all; clear all; clc;

%clustering parameters
Trials = 1000; %number trials used for consensus clustering
MaxK = 5; %max number consensus clusters
CountThreshold = 2000; %used to filter out cases with few nuclei

%load HPC pipeline results
[hNormMean hNormStd hMeans hStds hCounts hNames] = HPCAggregateResults();

%load Matlab results
load NormalizationParameters;
load GaussianModels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare normalization parameters and visualize %%%%%%%%%%%%%%%%%%%%%%%%%%

%calculate percent difference
pdNormMean = 100 * abs(NormMean - hNormMean) ./ max(abs(NormMean), abs(hNormMean));
pdNormStd = 100 * abs(NormStd - hNormStd) ./ max(abs(NormStd), abs(hNormStd));
figure; 
subplot(1,2,1); plot(pdNormMean, 'bo'); xlim([0 74]);
xlabel('Feature Index'); ylabel('Percent Difference'); title('Global Mean');
subplot(1,2,2); plot(pdNormStd, 'ro'); xlim([0 74]);
xlabel('Feature Index'); ylabel('Percent Difference'); title('Global Stdev');

%output as list
Output = cell(75, 3);
Output(1,:) = {'Feature Name', '% Diff. Mean', '% Diff. Stdev'};
Output(2:end,1) = FeatureNames(true);
Output(2:end,2) = cellfun(@(x)num2str(x), mat2cell(pdNormMean, ones(74,1)), 'UniformOutput', false);
Output(2:end,3) = cellfun(@(x)num2str(x), mat2cell(pdNormStd, ones(74,1)), 'UniformOutput', false);
cell2text(Output, 'GlobalMeanComparison.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare patient means and standard deviations and visualize %%%%%%%%%%%%%

%normalize 'hMeans' ('Means' is already normalized as appears in "GaussianModels.mat")
hMeans = (hMeans - hNormMean * ones(1,size(hMeans,2))) ./ ...
                    (hNormStd * ones(1,size(Means,2)));

%boxplot of differences in patient signatures, normalize both by matlab global stdev
figure; boxplot(Means.' - hMeans.', 'orientation', 'horizontal',...
                'labels', FeatureNames(true));
figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cluster analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load feature selection file
load SelectedFeatures;

%perform feature selection
Means = Means(Selected,:);
hMeans = hMeans(Selected,:);

%remove feature 13
Means(54,:) = [];
hMeans(54,:) = [];

%normalize together - should also investigatve normalizing separately
Normalized = quantilenorm([Means hMeans].').';
NormMeans = Normalized(:,1:size(Means,2));
hNormMeans = Normalized(:,size(Means,2)+1:end);

%cluster signatures for each dataset
[Labels Cophenet Ordered Indices Initializations Ks] = ...
            KMeansConsensusClustering(NormMeans, 3, Trials);
[hLabels hCophenet hOrdered hIndices hInitializations hKs] = ...
            KMeansConsensusClustering(hNormMeans, 3, Trials);        
        
%initialize containers for clustering quality metrics
Silhouettes = zeros(1,size(Labels,2));
hSilhouettes = zeros(1,size(hLabels,2));

%set K = 3.
Labels = Labels(:, Ks == 3);
hLabels = hLabels(:, hKs == 3);
Ordered = Ordered(:,:, Ks == 3);
hOrdered = hOrdered(:,:, hKs == 3);
Indices = Indices(:, Ks == 3);
Indices = hIndices(:, hKs == 3);

%plot consensus matrix
figure; PlotConsensusMatrix(Ordered, Labels, Indices, 'k:'); title('Matlab Dataset');
figure; PlotConsensusMatrix(hOrdered, hLabels, hIndices, 'k:'); title('HPC Dataset');

%plot silhouette profile, calculate silhouette sum
figure; [S h] = silhouette(NormMeans.', Labels); title('Matlab Dataset');
figure; [S h] = silhouette(hNormMeans.', hLabels); title('HPC Dataset');

%display clustergram
[Order LeafOrder] = Clustergram(Normalized, Labels); title('Matlab Dataset');
HFClustergram(hNormMeans, hLabels, LeafOrder); title('HPC Dataset');

%map patient lists
Mapping = StringMatch(hNames, Names);

%re-order data
hLabels = hLabels([Mapping{:}]);

%calculate co-occurrence between labels
Cooccurrence = zeros(3);
for i = 1:length(hLabels)
    Cooccurrence(Labels(i), hLabels(i)) = Cooccurrence(Labels(i), hLabels(i)) + 1;
end
