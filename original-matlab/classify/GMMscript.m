close all; clear all; clc;

% %get class labels
% directory = '\\storage01.cci.emory.edu\vg2.repos.data\Annotations\Nuclei Classification\';
% AnnotationFiles = dir([directory '*.xml']);
% Annotation = AperioXMLRead([directory AnnotationFiles(1).name]);
% for j = 1:length(Annotation.Layers)
%     names{j} = Annotation.Layers(j).Description;
% end

%load data
load('./QueryRS.mat');

data = [data_val; data_tcga];
slides = data(:,1);
features = cell2mat(data(:,2:end));

features(ismember(features(:,1),[5]),:)=[];
labels = features(:,1);

%features = cell2mat(Annotations(:,3:end));
%labels = cell2mat(Annotations(:,2));

%discard small classes
%discard = (labels == 5) | (labels == 6);
%labels(discard) = [];
%features(discard,:) = [];

%normalize
features = zscore(features);

%get classes
classes = unique(labels);

%feature selection
w = L1FeatureSelection(features.', labels, 1, 1e-2, 2.5);

CC1 = zeros(6,6,100); rate1=zeros(100,1);
for r = 1:100
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Perform 5-fold cross-validations for a variety of classifiers
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %partition = cvpartition(labels, 'kfold', 5);
    partition = cs{r};
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %L1 least squares
    for i = 1:partition.NumTestSets
        train = training(partition, i);
        testing = test(partition, i);
        
        trainFeatures = features(train, :);
        trainLabels = labels(train);
        testFeatures = features(testing, :);
        testLabels = labels(testing);
        
        [idx(:,i) SCI(:,i) solutions] = L1Classification(trainFeatures.', trainLabels, testFeatures.', 1e-3);
        
        L1confusion(:,:,i) = confusionmat(testLabels, idx(:,i));
        L1accuracy(i) = sum(diag(L1confusion(:,:,i))) / sum(testing);
    end
    L1confusion = sum(L1confusion, 3);
    L1perclasserror = (sum(L1confusion,2) - diag(L1confusion)) ./ sum(L1confusion,2);
    
    CC1(:,:,r) = L1confusion*100;
    rate1(r) = sum(diag(L1confusion))/sum(L1confusion(:));
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GMM, no feature selection, no filtering
clear mus sigmas indexes Posteriors;

CC2 = zeros(6,6,100); rate2=zeros(100,1);
for r = 1:100
    partition = cs{r};
    
    for i = 1:partition.NumTestSets
        train = training(partition, i);
        testing = test(partition, i);
        
        trainFeatures = features(train,:);
        trainLabels = labels(train);
        testFeatures = features(testing, :);
        testLabels = labels(testing);
        
        for j = 1:length(classes)
            mus(j,:) = mean(trainFeatures(trainLabels == classes(j), :), 1);
            sigmas(:,:,j) = cov(trainFeatures(trainLabels == classes(j), :));
        end
        
        GMM = gmdistribution(mus, sigmas, ones(1, length(classes)));
        
        [idx, nlogl, P] = cluster(GMM, testFeatures);
        
        indexes(:,i) = classes(idx);
        Posteriors(:,i) = max(P,[],2);
        
        GMMconfusion(:,:,i) = confusionmat(testLabels, indexes(:,i));
    end
    GMMconfusion = sum(GMMconfusion, 3);
    GMMperclasserror = (sum(GMMconfusion,2) - diag(GMMconfusion)) ./ sum(GMMconfusion,2);
    
    CC2(:,:,r) = GMMconfusion*100;
    rate2(r) = sum(diag(GMMconfusion))/sum(GMMconfusion(:));
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GMM, feature selection, no filtering
clear mus sigmas train testing trainFeatures trainLabels testFeatures testLabels GMM indexes Posteriors;
hits = w > 1e-3;

CC3 = zeros(6,6,100); rate3=zeros(100,1);
for r = 1:100
    partition = cs{r};
    
    for i = 1:partition.NumTestSets
        train = training(partition, i);
        testing = test(partition, i);
        
        trainFeatures = features(train, hits);
        trainLabels = labels(train);
        testFeatures = features(testing, hits);
        testLabels = labels(testing);
        
        for j = 1:length(classes)
            mus(j,:) = mean(trainFeatures(trainLabels == classes(j), :), 1);
            sigmas(:,:,j) = cov(trainFeatures(trainLabels == classes(j), :));
        end
        
        GMM = gmdistribution(mus, sigmas, ones(1, length(classes)));
        
        [idx,nlogl,P] = cluster(GMM, testFeatures);
        
        indexes(:,i) = classes(idx);
        Posteriors(:,i) = max(P,[],2);
        
        GMMReducedconfusion(:,:,i) = confusionmat(testLabels(maxPosterior >= 0), indexes(:,i));
    end
    GMMReducedconfusion = sum(GMMReducedconfusion, 3);
    GMMReducedperclasserror = (sum(GMMReducedconfusion,2) - diag(GMMReducedconfusion)) ./ sum(GMMReducedconfusion,2);
    
    
    CC3(:,:,r) = GMMReducedconfusion*100;
    rate3(r) = sum(diag(GMMReducedconfusion))/sum(GMMReducedconfusion(:));
end
