function [Labels CophenetCorr Ordered Indices Initializations ks] = KMeansModelSelection(A, d, trials)
%Uses consensus clustering via k-means to perform a model selection, 
%determining the number of natural clusters within the data.
%inputs:
%A - matrix of feature vectors in columns.
%d - maximum number of clusters to calculate.
%trials - number of trials to use in consensus clustering.
%outputs:
%Labels - sample labels (for the columns of A).
%CophenetCorr - cophenet correlation coefficient for # clusters from 2:d to 
%               evaluate model selection results (higher is better).
%Ordered - consensus clustering visualization of cluster cohesion.
%Indices - ordering of rows/columns in Ordered.
%Initializations - ks-1 cell array containing the initialization info used 
%                  for kmeans clustering. Each element is a trials x ks(i) 
%                  array listing the sample indices used for cluster seeds.
%ks - number of clusters used in trials.

%initialize outputs
Ordered = zeros(size(A,2), size(A,2), d-2+1);
CophenetCorr = zeros(1,d-2+1);
Labels = zeros(size(A,2), d-2+1);
Indices = zeros(size(A,2), d-2+1);
Initializations = cell(1,d-2+1);
ks = [2:d];

%model selection
for k = 2:d
    
    %update console
    fprintf('Consensus clustering, K=%d, ', k);
    
    %start timer
    tic;
    
    %generate consensus matrix
    [Consensus Initializations{k-1}] = KMeansComputeConsensus(A, k, trials);
    
    %re-order consensus matrix to obtain final clustering
    [Labels(:,k-1) Ordered(:,:,k-1) CophenetCorr(k-1) Indices(:,k-1)] = KMeansConsensusReorder(Consensus, k);
    
    %update console
    fprintf('%g seconds.\n', toc);
    
end

function [Consensus Initialization] = KMeansComputeConsensus(A, k, trials)
%Computes consensus matrix from multiple trials of K-means clustering.  Each
%trial creates an adjacency matrix indicating which samples are grouped
%together in classes.  The average over these trials is the consensus.
%inputs:
%A - dxN matrix with d-dimensional samples in columns.
%k - number of models to use during clustering.
%trials - number of trials to repeat clustering.
%outputs:
%Consensus - normalized [0, 1] adjacency matrix indicating strenght of
%            co-classification between samples.

%initialize outputs
Consensus = zeros(size(A,2), size(A,2), trials);
Initialization = zeros(trials, k);

%perform 'trials' experiments, clustering the columns of A using KMeans
for i = 1:trials
    
    %generate seeds
    [~, Order] = sort(rand(1,size(A,2)));
    Order = Order(1:k);
    
    %record seeds
    Initialization(i,:) = Order;
    
    %cluster
    [labels] = kmeans(A.', k, 'EmptyAction', 'singleton',...
                    'start', A(:,Order).');
            
	%calculate co-occurrence
    Consensus(:,:,i) = KMeansBuildCooc(labels);
       
end

%normalize Cooc to calculate Consensus
Consensus = mean(Consensus, 3);

function Cooc = KMeansBuildCooc(labels)
%
    Cooc = zeros(length(labels));
    for i = 1:length(labels)
        Cooc(i,i:end) = (labels(i:end) == labels(i));
    end
    Cooc = Cooc + Cooc.' - eye(length(labels));

function [Labels Ordered CophenetCorr Indices] = KMeansConsensusReorder(C, k)
%re-orders consensus matrix to produce model-selection visualization array
%'Ordered', and set of consensus clustering labels 'Labels'.
%inputs:
%C - normalized concensus matrix produces over multiple iterations of
%    stochastic clustering.
%k - number of clusters for hierarchical clustering.
%outputs:
%Ordered - version of 'C' with rows, columns re-ordered to reveal clustered 
%          groups as sub-matrices of block diagonal.
%Indices - order of rows/columns in 'Ordered'

%Rearrange UT portion of 'C' as output from pdist.  Convert similarity to 
%distance. Copy once to avoid costly growing of array.
pdist = cell(1,size(C,2)-1);
for i = 1:size(C,2)-1
    pdist{i} = 1-C(i,i+1:end);
end
pdist = [pdist{:}];

links = linkage(pdist, 'average');

[dummy dummy Indices] = dendrogram(links, 0);
close(gcf); %close dendrogram.

CophenetCorr = cophenet(links, pdist);

Labels = cluster(links, 'maxclust', k);

Ordered = C(Indices, Indices);