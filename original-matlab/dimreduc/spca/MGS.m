function [Q,R] = MGS(A)
%MGS performs QR decomposition using Modified Gram Schmidt
%   follows Golub algorithm 5.2.5.
    [m, n] = size(A);
    R = zeros(n, n);
    Q = zeros(m, n);
    for k = 1:n
        v = A(:, k);
        R(k, k) = sqrt(dot(v, v));
        v = v / R(k, k);
%        for j = k+1:n
%            R(k, j) = v' * A(1:m,j);
%            A(1:m, j) = A(1:m,j) - v*R(k,j);
%        end
        R(k, k+1:n) = v' * A(:,k+1:n);
        A(:, k+1:n) = A(:,k+1:n) - v * R(k,k+1:n);
        Q(:, k) = v;
    end

end

