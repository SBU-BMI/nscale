function rate = payroll_fcn(data, group, subset) 
subset = sort(subset);
data = data(:,subset);

k = 5;
c=cvpartition(group,'kfold',k); 

% performance = zeros(k,3);
% for i = 1:k
%     xt = data(test(c,i),:);
%     yt = group(test(c,i),1);
%     xT = data(training(c,i),:);
%     yT = group(training(c,i),1);
%     
%     [class, err, pos, logp, coeff] = classify(xt,xT,yT,'quadratic','empirical');
%     
%     
%     max_pos = max(pos,[],2);
%     sort_max_pos = sort(max_pos);
%     cutoff = sort_max_pos(round(0.1*length(sort_max_pos)));
%     
%     
%     tf_uncertain = max(pos,[],2)<cutoff;
%     performance(i,1) = sum(class(~tf_uncertain) == yt(~tf_uncertain));
%     performance(i,2) = size(yt(~tf_uncertain),1);
%     performance(i,3) = sum(tf_uncertain)/size(tf,1)*100;
% end
% 
% correct_total = sum(performance);
% rate = correct_total(1)/correct_total(2);
% correct_total(3) = correct_total(3)/k*100;
%fprintf(['AVG Filtering Percentage: %f -- features: ',repmat('%d ',1,length(subset)) '\n'],correct_total(3),subset);

fun = @(xT,yT,xt,yt)(sum(yt==classify(xt,xT,yT,'quadratic','empirical')));
rate = sum(crossval(fun,data,group,'partition',c))...
      /sum(c.TestSize);