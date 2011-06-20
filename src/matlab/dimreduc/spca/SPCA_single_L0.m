function [ p, x, z, iters] = SPCA_single_L0( A, gamma, x)
%SPCA_single_L0 Compute the Nestorov single unit SPCA based on L0 penalty
%   input:  A - data matrix, pxn
%           gamma - sparsity controlling parameter, non-negative
%           x - initial iterate, in Sp.  can be computed using
%               find_max_column with normalize flag on.
%  output:  p - a locally optimal sparsity pattern
%           x, z - x and z occurred at p.
% following Algorithm 3 of Nestorov paper

% stopping criteria: 
%       1. (f(x_k+1)-f(x_k)) / f(x_k) < epsilon = 10^-4,
%           f(x) = sum over i (max((a_i'x)^2 - gamma), 0))
%       2. max number of iterations reached.
%     
    maxiter=2147483647;  % some large number
    epsilon = 1e-4;

    % initialize the intermediate variables for the first pass
    Ax = A'*x;  % a_i' * x
    ofcomp = (Ax .* Ax) - gamma;  % and objective function components
    % update the objective function value
    of = max(ofcomp, 0);
    of = sum(of, 1);
    
    for iters = 1:maxiter
        % compute [sign((a_i'x)^2 - gamma)]_+ * a_i'x  (if set as z, would 
        % need to be normalized)
        z = max(sign(ofcomp), 0) .* Ax;
        % compute z_i * a_i (or a_i * z_i), then summing over all columns of A.
        % this can be done as a mat-vec multiply. update x
        x = A*z;
        % normalize x
        x = x ./ norm(x);
        
        % update variables for the next iteration.
        Ax = A'*x;  % a_i'x
        ofcomp = (Ax .* Ax) - gamma;  % and objective function components
        % update the objective function value
        ofnew = max(ofcomp, 0);
        ofnew = sum(ofnew, 1);
        
        % and evaluate the objective function
        if (((ofnew - of) / of) < epsilon)
            % exit criteria met.  finish up and return
            % construct P:  p_i = 0 if (a_i'x)^2 > gamma, p_i = 1 otherwise
            p = (ofcomp <= 0);
            
            % and compute z with the current x.
            z = max(sign(ofcomp), 0) .* Ax;
            zn = sqrt(sum((z .* Ax), 1));
            if (size(zn == 0) > 0)
                p = p(:, zn ~= 0);
                z = z(:, zn ~= 0);
                zn = zn(zn~=0);
            end
            z = z ./ zn;
            % if zn = 0, then z should be 0
            z(isnan(z)) = 0;

            return;
        else
            % exit criteria not met, get ready for next iter
            of = ofnew;
        end
    end
    % if we get here, we have exhausted the iterations
    error('gone past maxiter');
end

