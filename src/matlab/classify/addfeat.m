% add one feature to an existing feature set

function [winner, Cost, newFeat] = addfeat(X, Y, selectioncriteria, X_k)
[n,d]=size(X);
N=length(X_k);

cost=zeros(d-N,N+2); % initialize costs table
for i=1:d-N 
    cost(i,2:N+1)=X_k;
end;

ind=1;
for i=1:d
    if (isempty(find(X_k == i)))
        cost(ind,N+2)=i;
        subset=cost(ind,2:end);
        cost(ind,1)=eval(selectioncriteria);
        ind=ind+1;
    end;%end if
end;%end k

sorted_cost = sortrows(cost,1);% sort the costs
winner = sorted_cost(end,2:end); % add the best feature to winners
Cost = sorted_cost(end,1); % new cost after add operation
newFeat=sorted_cost(end,end); % new added feat