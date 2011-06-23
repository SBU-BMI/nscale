function [ P, X, iters ] = SPCA_block_L1( A, gamma, X )
%SPCA_block_L1 Compute the Nestorov single unit SPCA based on L1 penalty
%   input:  A - data matrix, pxn
%           gamma - sparsity controlling parameter, non-negative
%           X - initial iterate, in S_m^p
%  output:  P - a locally optimal sparsity pattern
% following Algorithm 4 of Nestorov paper

    
% stopping criteria:  
%     
%       1. (f(x_k+1)-f(x_k)) / f(x_k) < epsilon = 10^-4,
%           f(x) = sum over j rows: 
%                    sum over i columns:
%                       max((abs(a_i'x_j) - gamma), 0)) ^2
%           note that this is the square of frobenius norm of the matrix 
%               max((abs(a_i'x_j) - gamma), 0))
%       2. max number of iterations reached.
%     
    maxiter=2147483647;  % some large number
    epsilon = 1e-4;

    % initialize the intermediate variables for the first pass
    AX = A'*X;  % a_i'x_j
    ofcomp = max((abs(AX) - gamma), 0);  % and objective function components
    % update the objective function value
    % square of frobenius norm
    %of = ofcomp .* ofcomp;
    %of = sum(sum(of));
    of = norm(ofcomp, 'fro');
    of = of * of;
    
    for iters = 1:maxiter
        % compute [|a_i'x_j| - gamma]_+ * sign(a_i'x_j)
        Z = ofcomp .* sign(AX);
        % compute y_ij * a_i, summing over all i. this can be done as a
        % mat-mat multiply. update X
        X = A*Z;
        % compute U factor of polar decomposition of matrix X
        % this is X(X'X)^(-1/2).
        X = polar_decomp(X);
         
        % update variables for the next iteration.
        AX = A'*X;  % a_i'x_j
        ofcomp = max((abs(AX) - gamma), 0);  % and objective function components
        % update the objective function value
        %ofnew = ofcomp .* ofcomp;
        %ofnew = sum(sum(ofnew));
        ofnew = norm(ofcomp, 'fro');
        ofnew = ofnew * ofnew;
    
        
        % and evaluate the objective function
        if (((ofnew - of) / of) < epsilon)
            % exit criteria met.  finish up and return
            % construct P:  p_ij = 0 if |a_i'xj| > gamma, p_ij = 1 otherwise
            P = (abs(AX) <= gamma);
            PN = sum(P, 1);
            PS = size(P, 1);
            P = P(:, PN ~= PS);
            return;
        else
            % exit criteria not met, get ready for next iter
            of = ofnew;
        end
    end
    % if we get here, we have exhausted the iterations
    error('gone past maxiter');


end

