function [ P, X, Z, iters ] = SPCA_block_L0( A, gamma, X )
%SPCA_block_L0 Compute the Nestorov single unit SPCA based on L0 penalty
%   input:  A - data matrix, pxn
%           gamma - sparsity controlling parameter, non-negative
%           x - initial iterate, in S_m^p
%  output:  P - a locally optimal sparsity pattern
%           X, Z - X and Z occuring at P.
% following Algorithm 5 of Nestorov paper
% since the max. penalty does not depend on z, and the stopping criteria
% maximizes essentially the variance matrix.
    
% stopping criteria: 
%       1. (f(x_k+1)-f(x_k)) / f(x_k) < epsilon = 10^-4,
%           f(x) = sum over j (sum over i (max((a_i'x)^2 - gamma), 0)))
%       2. max number of iterations reached.
%     
    maxiter=2147483647;  % some large number
    epsilon = 1e-4;

    % initialize the intermediate variables for the first pass
    AX = A'*X;  % a_i'x_j
    ofcomp = (AX .* AX) - gamma;  % and objective function components
    % update the objective function value
    of = max(ofcomp, 0);
    of = sum(sum(of, 1), 2);
    
    for iters = 1:maxiter        
        % compute [sign((a_i'x_j)^2 - gamma)]_+ * a_i'x_j
        Z = max(sign(ofcomp), 0) .* AX;
        % compute y_ij * a_i (or a_i * y_ij), summing over all columns of A.
        % this can be done as a mat-mat multiply. update X
        X = A*Z;
        % compute U factor of polar decomposition of matrix X
        % this is X(X'X)^(-1/2).
        X = polar_decomp(X);
        
        % update variables for the next iteration.
        AX = A'*X;  % a_i'x_j
        ofcomp = (AX .* AX) - gamma;  % and objective function components
        % update the objective function value
        ofnew = max(ofcomp, 0);
        ofnew = sum(sum(ofnew, 1), 2);
        
        % and evaluate the objective function
        if (((ofnew - of) / of) < epsilon)
            % exit criteria met.  finish up and return
            % construct P:  p_ij = 0 if (a_i'xj)^2 > gamma, p_ij = 1 otherwise
            P = (ofcomp <= 0);
            
            % and compute z with the current x.
            Z = max(sign(ofcomp), 0) .* AX;
            ZN = sqrt(sum((Z .* AX), 1));  % sum over all observations
            if (size(ZN == 0) > 0)
                P = P(:, ZN ~= 0);
                Z = Z(:, ZN ~= 0);
                ZN = ZN(ZN~=0);
            end
            Z = bsxfun(@rdivide, Z, ZN);
            % if ZN == 0, that means ofcomp is negative or 0,
            % so the whole column doesn't contribute. so NaN from ZN = 0 ->
            % Z = 0
            Z(isnan(Z)) = 0;
            return;
        else
            % exit criteria not met, get ready for next iter
            of = ofnew;
        end
    end
    % if we get here, we have exhausted the iterations
    error('gone past maxiter');
end

