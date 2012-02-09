function PlotConsensusMatrix(Ordered, Labels, Indices, LineSpec)
%Plots the consensus matrix 'Ordered' with cluster labels.
%inputs:
%Ordered - NxN consensus matrix, as obtained from ModelSelection functions.
%Labels - N-length vector of sample labels, as obtained from ModelSelection functions.
%Indices - N-length vector of row/column order of 'Ordered', as obtained from ModelSelection functions.
%LineSpec - description of lines used to delineate cluster boundaries.

%find order in which clusters are displayed
[Clusters Index] = unique(Labels(Indices), 'first');
[Order Clusters] = sort(Index);

%create scaled image of consensus matrix
imagesc(Ordered); hold on; axis equal tight;

%build box indices
for i = 1:length(Clusters)
    if(i == 1)
        bottom(i) = 1;
        top(i) = sum(ismember(Labels, Clusters(1:i)));
    else
        bottom(i) = sum(ismember(Labels, Clusters(1:i-1)));
        top(i) = sum(ismember(Labels, Clusters(1:i)));
    end
end

%draw boxes, print labels
for i = 1:length(Clusters)
    plot([bottom(i) top(i) top(i) bottom(i) bottom(i)],...
        [bottom(i) bottom(i) top(i) top(i) bottom(i)], LineSpec, 'LineWidth', 3);
    text(mean([bottom(i) top(i)]), -5, num2str(Clusters(i)),...
        'FontSize', 18, 'HorizontalAlignment', 'center');
end