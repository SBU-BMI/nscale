function [labels SCI solutions] = L1Classification(A, truth, test, epsilon)
%Perform L1 classification of test samples using
%Sparse-Representation-based Classificaiton (SRC).  Solves the system
%y=Ax+e, x = arg min ||x||_1, ||Ax-y||_2 < e
%inputs:
%A - m x n matrix with training samples in columns.
%truth - n-length vector with training sample labels.
%test - m x p matrix with test samples in columns.
%epsilon - bound for solution error norm (L2).
%outputs:
%labels - p-length vector with labels for test samples.
%SCI - p-length vector with sparsity concentration indexes for test samples.
%solutions - n x p array containing L1 solutions for visualization of sparsity.

n = size(A,2); %get number of training samples
m = size(A,1); %get dimensionality

norms = sqrt(sum(A.^2, 1)); %calculate training sample norms

A = A ./ repmat(norms, [m 1]); %project training samples onto unit hypersphere

[truth order] = sort(truth); %sort labels in increasing order

A = A(:,order); %reorder training samples for convenience

classes = unique(truth); %get class labels

k = length(classes); %get number of classes

indicators = zeros(n, k); %calculate class indicators
for i = 1:k
    indicators(:,i) = (truth == classes(i));
end

counts = sum(indicators, 1); %calculate counts for each group

residuals = zeros(k, 1); %initialize residuals
solutions = zeros(n, size(test,2)); %initialize solutions
SCI = zeros(1, size(test,2));

start = A.'*inv(A*A.'); %used to calculate starting point for L1 solution

for i = 1:size(test,2) %classify each test sample
    clc;
    fprintf('%s of %s\n.', num2str(i), num2str(size(test,2)));
    
    sample = test(:,i) / norm(test(:,i)); %normalize test sample
    
    x0 = l1qc_logbarrier(start * sample, A, [], sample, epsilon, 1e-3); %L1 solver
    
    solutions(:,i) = x0; %record solution
    
    for j = 1:k %compute residuals
        residuals(j) = norm(sample - A * (indicators(:,j) .* x0));
    end 
    
    SCI(i) = (k * max(indicators.' * abs(x0)) / norm(x0,1) - 1) / (k - 1);
    
    [dummy index] = min(residuals); %calculate class
    labels(i) = classes(index);
end