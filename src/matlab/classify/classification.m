clc; clear all;
addpath('../spca', '../svd');

%ouput mat file for query results
QueryRSFile = './QueryRS.mat';
class_name = {'Normal',...
    'Neoplastic Astrocyte','Neoplastic Oligodendrocyte',...
    'Reactive Endothelial','Microglia','Mitotic Figure',...
    'Reactive Astrocyte','Junk'};

%load query results for local file if there is any
if exist('data','var')==0
    load(QueryRSFile,'data_val','data_tcga');
end

data = [data_val; data_tcga];

%find number of samples for each class
slides = data(:,1);
data = cell2mat(data(:,2:end));
data(ismember(data(:,1),[5]),:)=[];  % 5th class is excluded - microglia

group = data(:,1);
features = data(:, 2:end);
clear data;  % to ensure that "group" and "features" are used
unique_class = unique(group);

for i = unique_class'
    fprintf('Class %s: %d \n',class_name{i},sum(group==i));
end


% No feature selection
%%f_ind = [1:size(features,2)];

%selected features with cytoplasmic features
f_ind = [4 2 58 41 8 64 67 50 48 1 33 47 9 22 39 62 51 16 24 25 74 6 14 68 34 29 3 21 63 17 56 69 57 65];

%all nuclear features
%f_ind(f_ind<24)=[];

%selected features WITHOUT cytoplasmic features
%f_ind = [5 6 9 8 15 4 2 22 14 1 10 16 7 11 19 3 20 17 12 23]; %no cytoplasmic features




%cross-fold validation partition
k = 5;
iterations = 100;
fun = @(xT,yT,xt,yt)(sum(yt==classify(xt,xT,yT,'quadratic','empirical')));




%% transform data - PCA and SPCA: zero-centered, PCA/SPCA/TJ-PCA,
% features:0-n, norm 0/1. gamma 0:0.1:2

transformTypes = {'PCA', 'SPCA', 'TJ PCA'};

rate=zeros(iterations,1);
for Lnorm = 0:1
    for gamma = 4:-0.2:0
        
        featSize = size(features, 2);
        %[tdata P] = transformDataTry(features, group, 3, 2, featSize, Lnorm, gamma);        
        [tdata P] = transformData(features, 1, 2, featSize, Lnorm, gamma);        
        featSizeMax = size(P,2);
        clear P;
        
        for featSize = featSizeMax : -1 : 1
            %[tdata P] = transformDataTry(features, group, 3, 2, featSize, Lnorm, gamma);
            [tdata P] = transformData(features, 1, 2, featSize, Lnorm, gamma);
                        
            for r=1:iterations
                c=cvpartition(group,'kfold',k);
                %--compute cross-validation performance
                rate(r,1) = sum(crossval(fun,tdata,group,'partition',c))...
                    /sum(c.TestSize);
                clear c;
            end
            sparsity = sum(sum(P==1))/(size(P,1) * size(P,2));
            fprintf('run: zero centered, transform L%d SPCA, %d featues requested, gamma %f.  %d feasible features, Sparsity %f, Accuracy: %f \n', ...
                Lnorm, featSize, gamma, size(P, 2), sparsity, sum(rate,1)/size(rate,1));
            clear tdata;
            clear P;
        end
        
    end
end

%% standard PCA
%tdata = transformDataTry(features, group, 3, 1, -1, 0, 0);
tdata = transformData(features, 1, 1, -1, 0, 0);
for featSize = size(features, 2) : -1 : 1

    for r=1:iterations
        c=cvpartition(group,'kfold',k);
        %--compute cross-validation performance
        rate(r,1) = sum(crossval(fun,tdata(:, 1:featSize),group,'partition',c))...
            /sum(c.TestSize);
        clear c;
    end
    fprintf(['run: zero centered, transform PCA, %d featues, Accuracy: %f \n'], ...
        featSize, sum(rate,1)/size(rate,1));

end
clear tdata;

%% Tony and Jack's SVD implementation
tdata = transformData(features, 1, 3, -1, 0, 0);
for featSize = size(features, 2) : -1 : 1

    for r=1:iterations
        c=cvpartition(group,'kfold',k);
        %--compute cross-validation performance
        rate(r,1) = sum(crossval(fun,tdata(:, 1:featSize),group,'partition',c))...
            /sum(c.TestSize);
        clear c;
    end
    fprintf(['run: zero centered, transform TJ-PCA, %d featues, Accuracy: %f \n'], ...
        featSize, sum(rate,1)/size(rate,1));

end
clear tdata;



%% transform data - original jun's: zscore, no transform, all features
tdata = transformData(features, 2, 0, [1:size(features,2)], 0, 0);
%%data_norm = zscore(features);

rate=zeros(iterations,1);
CC=zeros(6,6,iterations);

for r=1:iterations
    c=cvpartition(group,'kfold',k);    
    %--compute cross-validation performance
    rate(r,1) = sum(crossval(fun,tdata,group,'partition',c))...
           /sum(c.TestSize);
    clear c;
end
fprintf(['zscore normalize, no transform, all featues.  Accuracy: %f \n'],sum(rate,1)/size(rate,1));

clear tdata;

%% transform data - original jun's: zscore, no transform, select features
tdata = transformData(features, 2, 0, f_ind, 0, 0);
%%data_norm = zscore(features);

rate=zeros(iterations,1); CC=zeros(6,6,iterations);

cs=cell(1,iterations);
for r=1:iterations
    c=cvpartition(group,'kfold',k);

    cs{1,r}=c;     
%     %compute confusion matrix
%     cv = zeros(k,2);
%     for i = 1:k
%         xt = tdata(test(c,i));
%         xT = tdata(training(c,i));
%         yT = group(training(c,i),1);
%         yt = group(test(c,i),1);
%         
%         [class, err, pos, logp, coeff] = classify(xt,xT,yT,'quadratic','empirical');
%         
%         %filtering
%         %max_pos = max(pos,[],2);
%         %sort_max_pos = sort(max_pos);
%         %cutoff = sort_max_pos(round(0.1*length(sort_max_pos)));
%         
%         %no filtering
%         cutoff = 0;
%         
%         tf_uncertain = max(pos,[],2)<cutoff;
%         fprintf('Filtering percentage: %f \n',sum(tf_uncertain)/size(tf_uncertain,1)*iterations);
%         
%         
%         cv(i,1)=sum(class(~tf_uncertain) == yt(~tf_uncertain));
%         cv(i,2)=size(yt(~tf_uncertain),1);
%         C(:,:,i)=confusionmat(yt(~tf_uncertain),class(~tf_uncertain));
%     end
%     rate(r,1) = sum(cv(:,1))/sum(cv(:,2));
%     
%     C=sum(C,3);
%     for i = 1:size(C,1)
%         C(i,:) = C(i,:)/sum(C(i,:));
%     end
%     C=C*iterations;
%     
%     CC(:,:,r)=C;
    
    
    %--compute cross-validation performance
%    fun = @(xT,yT,xt,yt)(sum(yt==classify(xt,xT,yT,'quadratic','empirical')));
    rate(r,1) = sum(crossval(fun,tdata,group,'partition',c))...
           /sum(c.TestSize);
    clear c;
end
fprintf(['zscore normalize, no transform, SFFS selected best featues.  Accuracy: %f \n'],sum(rate,1)/size(rate,1));
clear tdata;

%% feature selection using SFFS
% NUM_feature = size(data_norm,2);
% 
% for i = 2:NUM_feature
%     selectioncriteria = 'payroll_fcn(X, Y, subset)';
%     [winner, accuracy, winners, accuracies] =...
%         SFFS(data_norm, group, i, 1, selectioncriteria);
%     fprintf(['Accuracy: %f -- features: ',repmat('%d ',1,length(winner)) '\n'],accuracy,winner);
% end