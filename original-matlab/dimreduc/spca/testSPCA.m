s0 = RandStream.create('mrg32k3a','seed', 0, 'NumStreams',1);
AA = randn(s0,20,5);
avg = sum(AA, 1) / size(AA,1);
A = bsxfun(@minus, AA, avg);

% calculate the principle components of A
[U D V] = svd(A);

disp('original variance');
var = trace(A'*A)

for k = 1:5
    z = V(:, 1:k);
    disp(sprintf('variance of PCA, with %d component only', k));
    var = adjusted_var(A, z)
end


% SPCA
% choose gamma to be 1
gamma = 1;

% single, L1
x = initializeX(A, 1);
[p, x2, iters] = SPCA_single_L1(A, gamma, x);
[x, z] = loading_single_L1(A, p, 1, x);
disp('variance of SPCA, with single component L1, gamma=1');
z
var = adjusted_var(A, z)
iters

% block, L1
for k = 2:5
    X = initializeX(A,k);
    [P, X2, iters] = SPCA_block_L1(A, gamma, X);
    [X, Z, iters2] = loading_block_L1(A, P, 1, X);
    disp(sprintf('variance of SPCA, with block %d components L1, gamma=1', k));
    var = adjusted_var(A, Z)
    iters
    iters2
end

% single, L0
x = initializeX(A, 1);
[p, x, z, iters] = SPCA_single_L0(A, gamma, x);
disp('variance of SPCA, with single component L0, gamma=1');
z
var = adjusted_var(A, z)
iters

% block, L0
for k = 2:5
    X = initializeX(A,k);
    [P, X, Z, iters] = SPCA_block_L0(A, gamma, X);
    disp(sprintf('variance of SPCA, with block %d components L0, gamma=1', k));
    var = adjusted_var(A, Z)
    iters
end
