function [ x, z ] = loading_single_L1( A, p, N, x )
%loading_single_L1 compute the loading vector z for the selected
%components to maximize variance explained.
%  Input:   A - data
%           p - sparsity pattern
%           N - a weighting to force separation of the components.  use I
%           x - initial x
% Output:   x - computed x to maximize variance explained
%           z - computed loading vector to maximize variance explained.
    
    % get the active entries of A.  recall p defines sparsity, i.e. entries
    % not active
    Anp = A(:, not(p));
    [U, D, V] = svd(Anp);
    
    % the solution is the rank one SVD of Anp.  Since we are working with
    % single-unit, m = 1, we can choose the right and left eigenvectors
    % corresponding to the largest eigenvalue (singular value).
    % x = u_1, z_active = v_1, and z_p = 0.
    x = U(:, 1);
    z = zeros(size(A,2), 1);
    z(not(p)) = V(:, 1);  % already normalized.
end

