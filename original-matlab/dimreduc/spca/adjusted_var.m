function [ V ] = adjusted_var( A, Z )
%adjusted_var Adjust the variance matrix using the sparse loading vectors
%  Input:   A - data matrix
%           Z - loading vectors
% Output:   V - variance 
% for single component, the adjusted variance is just the variance.
% for multiple compoentns, the adjusted variance is as described in 
% Nestorov, as adopted from Zou.

    [m, n] = size(Z);
    if (n == 1)
        V = A*Z;
        V = V' * V;
    else
        % compute the adjusted var
        % first compute AZ
        Y = A * Z;
        
        % next compute the QR decomposition of Y.
        [Q R] = MGS(Y);
        
        V = trace(R*R);
    end

end

