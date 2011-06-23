function w = L1FeatureSelection(X, labels, sigma, tau, lambda)
%inputs:
%X - J x N matrix with J-dimensional feature vectors in columns.
%labels - N-length array with class labels
%sigma - exponential kernel parameter k(d) = exp(-d/sigma).
%tau - stopping criterion for iteration |w_last - w| <= tau.
%lambda - regularization weight for optimization process (lambda * ||w||
%         penalty
%outputs:
%w - feature selection weights, should be nearly sparse.

[J N] = size(X); %get dimension and number of samples

Differences = PairwiseDifferences(X); %calculate pairwise sample differences

Hn = CalculateHn(labels); %calcualte sample sets
Mn = CalculateMn(labels);

w = ones(J, 1); %initialize w, w_last, bias
w_last = zeros(J, 1);

zn = zeros(J,N); %initialize z_n 

iteration = 1;
while(norm(w-w_last) >= tau)
    wD = sqrt(reshape(w.' * abs(Differences.^2), [N N]).'); %update weighted distance norm
    
    kwD = exp(-wD / sigma); %calculate kernel distances
    
    PNM = CalculatePNM(kwD, Mn); %calculate P(x_i = NM(x_n)|w)
    PNH = CalculatePNH(kwD, Hn); %calculate P(x_i = NH(x_n)|w)
    
    for i = 1:N %calculate \bar{z}_n
        zn(:,i) = abs(Differences(:, find(Mn(i,:)) + (i-1) * N)) * PNM{i}.' ...
            - abs(Differences(:, find(Hn(i,:)) + (i-1) * N)) * PNH{i}.';
    end
    
    w_last = w;
    
    v = sqrt(w);
    minfunc = @(x)L1FeatureSelectionObjective(x, zn, lambda);
    
    %options = optimset('GradObj','on');
    %v = fminunc(minfunc, v, options);
    v = fminunc(minfunc, v);
    w = v.^2;
    w = w / max(w);
    
    figure; stem(w); set(gca,'xscale','log');
end

w = w / max(w);

end

function vt = SearchDirection(v, zn, lambda)
    numerator = exp(- (v.^2).' * zn);
    vt = v.*(lambda*ones(length(v),1) - zn * (numerator ./ (1+numerator)).');
end

function PNM = CalculatePNM(kwD, Mn)
%Calculate probabilities P(x_i = NM(x_n)|w), i \in Mn for n = 1,...,N
%inputs:
%kwD - N x N matrix of kernel-transformed weighted distances.
%Mn - N x N indicator matrix with row/column n corresponding to the set M_n

    PNM = cell(1,size(Mn,1));

    for i = 1:size(Mn,1) %calculate denominators
        PNM{i} = kwD(i, Mn(i,:)) / (kwD(i,:) * Mn(i,:).');
    end
end

function PNH = CalculatePNH(kwD, Hn)
%Calculate probabilities P(x_i = NH(x_n)|w), i \in Hn for n = 1,...,N
%inputs:
%kwD - N x N matrix of kernel-transformed weighted distances.
%Hn - N x N indicator matrix with row/column n corresponding to the set H_n

    PNH = cell(1,size(Hn,1));

    for i = 1:size(Hn,1) %calculate denominators
        PNH{i} = kwD(i, Hn(i,:)) / (kwD(i,:) * Hn(i,:).');
    end    
end

function Hn = CalculateHn(labels)
%calculate sets of samples with same label as sample 'n'.
%inputs:
%labels - N-length matrix of class labels.
%outputs:
%Hn - NxN indicator matrix with Hn in row/column n.

Hn = false(length(labels)); %initialize output

for i = 1:length(labels)
    Hn(i,:) = (labels == labels(i));
    Hn(i,i) = false; %remove self from set.
end

end

function Mn = CalculateMn(labels)
%calculate sets of samples with different labels than sample 'n'.
%inputs:
%labels - N-length matrix of class labels.
%outputs:
%Mn - NxN indicator matrix with Hn in row/column n.

Mn = false(length(labels)); %initialize output

for i = 1:length(labels)
    Mn(i,:) = ~(labels == labels(i));
end

end
   
function Differences = PairwiseDifferences(X)
    Differences = cell(1,size(X,2));
    for i = 1:size(X,2)
        Differences{i} = X - X(:,i) * ones(1,size(X,2));
    end
    
    Differences = [Differences{:}];
end

% function plotdata(data, labels, PNM, Mn, PNH, Hn, n)
%     plot(data(1,labels==1), data(2,labels==1), 'bo'); hold on;
%     plot(data(1,labels==-1), data(2,labels==-1), 'rd');
%     plot(data(1,n), data(2,n), 'kx', 'MarkerSize', 14, 'LineWidth', 3);
%     axis equal tight;
%     colormap(jet);
%     cmap = colormap;
%     colorbar;
%     
%     hits = find(Hn(n,:));
%     if(labels(n) == 1)
%         for i = 1:length(hits);
%             plot(data(1,hits(i)), data(2,hits(i)), 'o', 'MarkerFaceColor', cmap(ceil((PNH{n}(i)/max(PNH{n})) * size(cmap,1)),:));
%         end
%     else
%         for i = 1:length(hits);
%             plot(data(1,hits(i)), data(2,hits(i)), 'd', 'MarkerFaceColor', cmap(ceil((PNH{n}(i)/max(PNH{n})) * size(cmap,1)),:));
%         end
%     end
% end