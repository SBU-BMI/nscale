function [Order LeafOrder] = Clustergram(SummaryStats, Labels)
%Create clustergram heatmap using hierarchical clustering to organize samples.
%inputs:
%SummaryStats - MxN matrix of normalized morphological signatures with 
%               features in rows and samples in columns.
%Labels - N-length vector of sample labels.
%outputs:
%Order - N-length vector of sample order, as they appear in clustergram.
%LeafOrder - M-length vector of feature order, as they appear in clustergram.

%order samples by label
[Sorted Order] = sort(Labels);

%hierarchical clustering of rows (to group features with similar values)
Distances = pdist(SummaryStats, 'euclidean');
Linkage = linkage(Distances, 'average');
if(max(size(SummaryStats)) <= 1000)
    LeafOrder = optimalleaforder(Linkage, Distances);
else
    [dummy dummy LeafOrder] = dendrogram(Linkage, 0);
end
    
%show clustergram
figure; image(SummaryStats(LeafOrder, Order), 'CDataMapping', 'Scaled');
colormap(redgreencmap(128));
set(gca, 'Clim', [-3 3]);
hold on;

%show cluster boundaries on clustergram
for i = 1:max(Labels)-1
    plot([sum(Labels <= i) sum(Labels <= i)] + 1/2,...
        [1 size(SummaryStats,1)], 'c', 'LineWidth', 2);
end

%label axes and remove x-ticks and labels
ylabel('Feature Indices');
set(gca, 'XTick', []);