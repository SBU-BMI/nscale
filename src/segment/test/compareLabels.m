function [ count1 count2 cooccur ] = compareLabels( label1, label2 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    count1 = length(unique(label1));
    count2 = length(unique(label2));
    
    cooccur = zeros(count1, count2, 'int32');
    
    x = reshape(label1, size(label1, 1) * size(label1, 2), 1);
    x = x+1;
    y = reshape(label2, size(label2, 1) * size(label2, 2), 1);
    y = y + 1;

    for i = 1 : length(x)
        cooccur(x(i, 1), y(i, 1)) = 1;
    end
    
end

