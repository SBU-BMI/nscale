function [ X, Z, iters ] = loading_block_L1( A, P, N, X )
%loading_block_L1 using the sparsity pattern, compute the loading vector
%and X to maximize variance
%   Input:  A - data matrix
%           P - sparsity pattern.  note that if a column is all sparse,
%           that column is removed in SPCA_block_L1.m.
%           If a whole column is 0, that should be excluded.
%           N - diag matrix with mu_i.  set to I
%           X - initial iterate
%  Output:  X - maximizes Tr(X'AZN) with Z
%           Z - maximizes Tr(X'AZN) with X

% N is set to I according to the paper (gamma parameter enforces the 
% stopping criteria:
%   1. (f(x_k+1)-f(x_k)) / f(x_k) < epsilon = 10^-4
%           f(x) = Tr(X'AZN)
%       2. max number of iterations reached.
%     
    maxiter=2147483647;  % some large number
    epsilon = 1e-4;

    % initialize the intermediate variables for the first pass
    % Specifically, compute Z.
    AX = A'*X;
    Z = AX;
    Z(P) = 0;
    d = diag(Z'*Z);  % diag(X'AA'X)  N = I. 
    % maximize Z.  from AX and diag(AX' *AX) computed previously
    % get Z Diag(Z'*Z)^-1/2
    d = 1 ./ sqrt(d);
    Z = Z * diag(d);
    
    of = sum(diag(AX' * Z), 1);     % objective function = Tr(X'AZN)
    
    for iters = 1:maxiter
        %  maximize X given new Z
        X = polar_decomp(A*Z); % AZN, but N = I

        % compute new Z given new X.
        AX = A' * X;
        Z = AX;
        % zero some entries 
        Z(P) = 0;
        
        d = diag(Z'*Z);  % diag(X'AA'X)  N = I. 
        % maximize Z.  from AX and diag(AX' *AX) computed previously
        % get Z Diag(Z'*Z)^-1/2
        d = 1 ./ sqrt(d);
        Z = Z * diag(d);
        
        % now compute the new obj function: = Tr(X'AZN)
        ofnew = sum(diag(AX'*Z), 1);
        
        % evaluate the objective function
        if (((ofnew - of) / of) < epsilon)
            % exit criteria met.
            return;
        else 
            % exit criteria not met, get ready for next iter
            of = ofnew;
        end        
    end
    % if we get here, we have exhausted the iterations
    error('gone past maxiter')

end

