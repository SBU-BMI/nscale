function [ U ] = polar_decomp( X )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    % compute U factor of polar decomposition of matrix X
    % this is X(X'X)^(-1/2).
    
    % this is don via SVD:  UDV' = X, X(X'X)^-1/2 = 
    % UDV'(VD'U'UDV')^-1/2 = UDV'(VD^2V')-1/2 = UDV'(VD
    
    [U, D, V] = svd(X);
    [m n] = size(D);
    U = U(:, 1:n) * V';
end

