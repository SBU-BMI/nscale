function [ X, ID ] = initializeX( A, blocksize )
%initializeX choose the appropriate X for the maximization of the objective 
% function during SPCA
%  Input:   A - data matrix
%           blocksize - number of vectors to include
% Output:   X - the chosen vectors
%           ID - the vectors chosen out of the original A matrix

    if (blocksize == 1) 
        [X, ID] = find_max_column(A, 1);
    else
        [X, ID] = MGS_with_norm_sort(A);
        X = X(:, 1:blocksize);
        ID = ID(1:blocksize);
    end
end

