function [ p, x, iters ] = SPCA_single_L1( A, gamma, x )
%SPCA_single_L1 Compute the Nestorov single unit SPCA based on L1 penalty
%   input:  A - data matrix, pxn
%           gamma - sparsity controlling parameter, non-negative
%           x - initial iterate, in Sp.  can be computed using
%               find_max_column with normalize flag on.
%  output:  p - a locally optimal sparsity pattern
% following Algorithm 2 of Nestorov paper

% stopping criteria. 
%       1. (f(x_k+1)-f(x_k)) / f(x_k) < epsilon = 10^-4,
%           f(x) = sum over i columns: max((abs(a_i'x) - gamma), 0)) ^2
%       2. max number of iterations reached.
%     
    maxiter=2147483647;  % some large number
    epsilon = 1e-4;

    % initialize the intermediate variables for the first pass
    Ax = A'*x;  % a_i' * x
    ofcomp = max((abs(Ax) - gamma), 0);  % and objective function components
    of = ofcomp .* ofcomp;
    % update the objective function value
    of = sum(of, 1);
    
    for iters = 1:maxiter
        % compute [|a_i'x| - gamma]_+ * sign(a_i'x)
        z = ofcomp .* sign(Ax);
        % compute z_i * a_i, summing over all i. this can be done as a
        % mat-vec multiply. update x
        x = A*z;
        % normalize x
        x = x ./ norm(x);

        
        % update variables for the next iteration.
        Ax = A'*x;  % a_i'x
        ofcomp = max((abs(Ax) - gamma), 0);  % and objective function components
        ofnew = ofcomp .* ofcomp;
        % update the objective function value
        ofnew = sum(ofnew, 1);
        
        % and evaluate the objective function
        if (((ofnew - of) / of) < epsilon)
            % exit criteria met.  finish up and return
            % construct P:  p_i = 0 if |a_i'x| > gamma, p_i = 1 otherwise
            p = (abs(Ax) <= gamma);
            pn = sum(p, 1);
            ps = size(p, 1);
            p = p(:, pn ~= ps);
            return;
        else
            % exit criteria not met, get ready for next iter
            of = ofnew;
        end
    end
    % if we get here, we have exhausted the iterations
    error('gone past maxiter');
end

