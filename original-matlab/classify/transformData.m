function [ newD P ] = transformData( D, normalize, transform, features, Lnorm, gamma )
%transformData perform data transformation before further classification
%   Input:  D - original data.  rows are samples, columns are features.
%           normalize - normalize using one of three strategies
%               0 - not normalized
%               1 - zero-center the data
%               2 - (zScore) zero-center the data and scale by std-dev
%           transform - types of transform to use, one the the following
%               0 - no transform
%               1 - PCA, SVD implementation
%               2 - SPCA, Nestorov formulation.
%               3 - PCA, Tony and Jack's SVD implementation
%           features - the features to include in the output
%               -1 - all the features.  applies to no transform, PCA, and
%                   SPCA, and PCA with Tony's SVD.
%               v - v is a vector indicating the indices of the feature
%                       columns to select.  This is to use with transform=0
%                       may be ordered.
%               n - n is a number indicating the number of features to keep
%                       when transform=1 or 3, n indicate the first n PC
%                       when transform=2, n indicate the SPCA block size
%           Lnorm - only used by SPCA. either 0 for L0 norm, or 1 for L1
%               norm constrained optimization
%           gamma - only used by SPCA.  this is a parameter to control
%               sparsity of the principle components.


% first normalize
    newD = D;
    P=1;
    [m, n] = size(newD);
    if (normalize == 1)
        newD = bsxfun(@minus, newD, sum(newD, 1)/m);
    elseif (normalize == 2)
        newD = zscore(newD);
    end
    % if normalize = 0 - no operation required.
    
    % next transform
    if (transform == 0)
        % no transform
        if (features == -1) 
            % and no selection of features, then nothing further to do
            return;
        else
            % select the features indicated in the vector
            newD = newD(:, features);
        end
    elseif (transform == 1)
        % standard matlab SVD implementation.  V matrix has the principle
        % components (VSSV' = A'A)
        [U, S, V] = svd(newD);
        clear U;
        clear S;
        if (features > 0)
            V = V(:, 1:features);
        end
        newD = newD * V;
    elseif (transform == 3)
        % Tony and Jack's matlab implementation.  V matrix has the principle
        % components (VSSV' = A'A)
        [U, S, V] = svdfinal(newD);
        clear U;
        clear S;
        if (features > 0)
            V = V(:, 1:features);
        end
        newD = newD * V;
    elseif (transform == 2)
        % Nestorov SPCA implementation
        % compute the loading
        if (Lnorm == 0)
            if (features < 2)
                X = initializeX(newD, 1);
                [P, X, Z, iters] = SPCA_single_L0(newD, gamma, X);
            else
                X = initializeX(newD, features);
                [P, X, Z, iters] = SPCA_block_L0(newD, gamma, X);
            end
        elseif (Lnorm == 1)
            if (features < 2)
                X = initializeX(newD, 1);

                [P, X2, iters] = SPCA_single_L1(newD, gamma, X);
                [X, Z] = loading_single_L1(newD, P, 1, X);
            else
                X = initializeX(newD, features);

                [P, X2, iters] = SPCA_block_L1(newD, gamma, X);
                [X, Z, iters2] = loading_block_L1(newD, P, 1, X(:, 1:size(P,2)));
            end
        end
        clear X;
        newD = newD * Z;
        clear Z;
    else
        error('ERROR.  Unsupported transform type');
    end
end

