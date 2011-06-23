function [winner, Cost] = SFS(X, Y, new_size, selectioncriteria)

[n,d]=size(X);

cost=zeros(d,2);%initialize costs
cost(:,2)=[1:d]';%label costs
for(i=1:length(cost)) 
    % Form costs for each individual feature  
    subset=cost(i,2);
    cost(i,1) = eval(selectioncriteria);
end
sorted_cost = sortrows(cost,1);% sort the costs
winner =  sorted_cost(d,2);% pull the best feature

Cost(1)=sorted_cost(d,1);

%repeat until selecting required # features
for i=2:new_size   
    j=i-1; % think the first index as 0 to make life easier
    cost=zeros(d-j,i+1);
    % insert the previous winners
    for k=1:d-j 
        cost(k,2:i)=winner;
    end;
    ind=1;
    
    %compute the next winner
    for k=1:d
        if (isempty(find(winner == k)))
            cost(ind,i+1)=k;
            subset=cost(ind,2:end);
            cost(ind,1)=eval(selectioncriteria);
            ind=ind+1;
        end;%end if
    end;%end k
    
    sorted_cost = sortrows(cost,1);% sort the costs
    winner =[winner sorted_cost(d-j,i+1)];% add the best feature to winners
    
    Cost(i)=sorted_cost(d-j,1);
end;%end i
Cost=Cost';