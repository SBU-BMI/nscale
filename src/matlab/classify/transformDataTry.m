function [ newD P ] = transformDataTry( D, C, normalize, transform, features, Lnorm, gamma )
%transformDataTry perform data transformation before further classification
%   Input:  D - original data.  rows are samples, columns are features.
%           C - sample classification.
%           normalize - normalize using one of three strategies
%               0 - not normalized
%               1 - zero-center the data
%               2 - (zScore) zero-center the data and scale by std-dev
%               3 - first zscore per class, shift back to class mean, then
%               zero-center the data - just for PCA/SPCA.  D is centered
%               prior to return.
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
    classes = unique(C);
    
    if (normalize == 1)
        newD = bsxfun(@minus, newD, sum(newD, 1)/m);
        intD = newD;
    elseif (normalize == 2)
        newD = zscore(newD);
        intD = 0;
    elseif (normalize == 3)
        % scale each class by its variance, while keeping the center.
        intD = newD;
        for cl = classes'
            intD(C==cl, :) = bsxfun(@plus, zscore(intD(C==cl, :)), sum(intD(C==cl, :),1)/size(intD(C==cl, :),1));
        end
        % then center the data
        intD = bsxfun(@minus, intD, sum(intD, 1)/m);
        % also center the original.
        newD = bsxfun(@minus, newD, sum(newD, 1)/m);
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
        Z = 0;
    elseif (transform == 1)
        % standard matlab SVD implementation.  V matrix has the principle
        % components (VSSV' = A'A)
        [U, S, Z] = svd(intD);
        clear U;
        clear S;
        if (features > 0)
            Z = Z(:, 1:features);
        end
        newD = newD * Z;
    elseif (transform == 3)
        % Tony and Jack's matlab implementation.  V matrix has the principle
        % components (VSSV' = A'A)
        [U, S, Z] = svdfinal(intD);
        clear U;
        clear S;
        if (features > 0)
            Z = Z(:, 1:features);
        end
        newD = newD * Z;
    elseif (transform == 2)
        % Nestorov SPCA implementation
        % compute the loading
        if (Lnorm == 0)
            if (features < 2)
                X = initializeX(intD, 1);
                [P, X, Z, iters] = SPCA_single_L0(intD, gamma, X);
            else
                X = initializeX(intD, features);
                [P, X, Z, iters] = SPCA_block_L0(intD, gamma, X);
            end
        else (Lnorm == 1)
            if (features < 2)
                X = initializeX(intD, 1);

                [P, X2, iters] = SPCA_single_L1(intD, gamma, X);
                [X, Z] = loading_single_L1(intD, P, 1, X);
            else
                X = initializeX(intD, features);

                [P, X2, iters] = SPCA_block_L1(intD, gamma, X);
                [X, Z, iters2] = loading_block_L1(intD, P, 1, X(:, 1:size(P,2)));
            end
        end
        clear X;
        clear intD;
        newD = newD * Z;
    else
        error('ERROR.  Unsupported transform type');
    end
end

