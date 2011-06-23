function [ x id ] = find_max_column( A, normalize )
%find_max_column find the column with the max norm in A
%  Input:   A - input matrix.
% Output:   x - the column.

    n2 = dot(A, A, 1);  % instead of doing vector norm on the columns of A,
        % we do dot product.
    [m id] = max(n2);  % find the max
    
    id = id(1);  % in case there are multiple, get the first one.
    
    x = A(:, id);  % get the column with the max.
    
    if (normalize)
        x = x / norm(x);
    end
end

