function [ X ID ] = MGS_with_norm_sort( A )
%MGS_with_norm_sort Orthogonalize the columns of A, with sorting
%by column 2-norm during each iteration.
%   The purpose of the function is to generate a matrix with orthogonal
%   columns.  This is done via the Modified Gram-Schmidt process with an 
%   additional modification - choice of next column is by largest norm. 
%   During each iteration, a column is chosen (with largest norm).  the
%   remaining columns are made orthogonal to the chosen column.  the
%   remaining columns are made available for hte next iteration. 
%  Input:   A - data matrix
% Output:   X - orthogonal columns

    rA = A;
    n = size(rA, 2);
    X = zeros(size(rA));
    orig_id = 1:n;
    ID = zeros(n, 1);
    for i = 1 : n
        % get the current choice of vector
        [x, id] = find_max_column(rA, 1);
        X(:, i) = x;
        ID(i) = orig_id(id);  % store the original column id in its new position
        orig_id(id) = 0;
        orig_id = orig_id(logical(orig_id));  % remove the "0" entry.
        
        % get the remaining columns as a matrix
        remaining = true(size(rA, 2),1);
        remaining(id) = 0;
        rA = rA(:, remaining);

        % do one step of MGS to orthogonalize rA to x
        dt = x' * rA;  % 1x(n-i) vector
        rA = rA - x * dt;
    end
end

