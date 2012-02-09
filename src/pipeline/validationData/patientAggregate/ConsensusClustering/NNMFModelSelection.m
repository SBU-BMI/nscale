function [Labels CophenetCorr Ordered ks] = NNMFModelSelection(A, d, trials)
%Uses consensus clustering via non-negative matrix factorization to perform
%a model selection, determining the number of natural clusters within the
%data.
%inputs:
%A - matrix of feature vectors in columns.
%d - maximum number of clusters to calculate.
%trials - number of trials to use in consensus clustering.
%outputs:
%Labels - sample labels (for the columns of A).
%CophenetCorr - cophenet correlation coefficient for # clusters from 2:d to 
%               evaluate model selection results (higher is better).
%Ordered - consensus clustering visualization of cluster cohesion.
%ks - number of clusters used in trials.

if(min(A(:)) < 0)
    correction = -min(A(:));
else
    correction = 0;
end

%initialize outputs
Ordered = zeros(size(A,2), size(A,2), d-2+1);
CophenetCorr = zeros(1,d-2+1);
Labels = zeros(size(A,2), d-2+1);
ks = [2:d];

for k = 2:d
    Consensus = NNMFComputeConsensus(A+correction, k, trials);
    [Labels(:,k-1) Ordered(:,:,k-1) CophenetCorr(k-1)] = NNMFConsensusReorder(Consensus, k);
end

function Consensus = NNMFComputeConsensus(A, k, trials)
%Computes consensus matrix from multiple trials of NNMF clustering.  Each
%trial creates an adjacency matrix indicating which samples are grouped
%together in classes.  The average over these trials is the consensus.
%inputs:
%A - dxN matrix with d-dimensional samples in columns.
%k - number of models to use during clustering.
%trials - number of trials to repeat clustering.
%outputs:
%Consensus - normalized [0, 1] adjacency matrix indicating strenght of
%            co-classification between samples.

%perform 'trials' experiments, clustering the columns of A using NNMF
Cooc = zeros(size(A,2), size(A,2), trials);
for i = 1:trials
    [W H] = nnmf(A, k, 'algorithm', 'mult');
    [dummy labels] = min(H, [], 1);
    Cooc(:,:,i) = NNMFBuildCooc(labels);
end

%normalize Cooc to calculate Consensus
Consensus = mean(Cooc, 3);

function Cooc = NNMFBuildCooc(labels)
%
    Cooc = zeros(length(labels));
    for i = 1:length(labels)
        Cooc(i,i:end) = (labels(i:end) == labels(i));
    end
    Cooc = Cooc + Cooc.' - eye(length(labels));

function [Labels Ordered CophenetCorr] = NNMFConsensusReorder(C, k)
%re-orders consensus matrix to produce model-selection visualization array
%'Ordered', and set of consensus clustering labels 'Labels'.
%inputs:
%C - normalized concensus matrix produces over multiple iterations of
%    stochastic clustering.
%k - number of clusters for hierarchical clustering.
%outputs:
%Ordered - version of 'C' with rows, columns re-ordered to reveal clustered 
%          groups as sub-matrices of block diagonal.

%Rearrange UT portion of 'C' as output from pdist.  Convert similarity to 
%distance. Copy once to avoid costly growing of array.
pdist = cell(1,size(C,2)-1);
for i = 1:size(C,2)-1
    pdist{i} = 1-C(i,i+1:end);
end
pdist = [pdist{:}];

links = linkage(pdist, 'average');

[dummy dummy indices] = dendrogram(links, 0);
close(gcf); %close dendrogram.

CophenetCorr = cophenet(links, pdist);

Labels = cluster(links, 'maxclust', k);

Ordered = C(indices, indices);