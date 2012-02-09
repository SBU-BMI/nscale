function [Labels Cophenet Ordered Indices Initializations Ks] = ...
    ConsensusClustering(SummaryStats, MaxClusters, Iterations)
%Performs consensus clustering using K-means from K=2:MaxClusters.
%inputs:
%SummaryStats - MxN matrix of normalized morphological signatures.
%MaxClusters - scalar natural number >= 2 indicating the max number of
%              clusters to apply.
%Iterations - number of iterations for consensus at each value of K.
%outputs:
%Labels - NxKs matrix of consensus cluster labels for each value of K.
%Cophenet - Ks-length vector of cophenet correlations for each clustering.
%Ordered - NxNxKs consensus matrices.
%Indices - NxKs matrix of ordering of rows/columns in Ordered.
%Initializations - ks-1 cell array containing the initialization info used 
%                  for kmeans clustering. Each element is a trials x ks(i) 
%                  array listing the sample indices used for cluster seeds.
%Ks - Ks-length vector of K values.

%perform consensus clustering
[Labels Cophenet Ordered Indices Initializations Ks] = ...
    KMeansModelSelection(SummaryStats, MaxClusters, Iterations);