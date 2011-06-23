% remove one feature to an existing feature set
% the one with least effect
function [winner, Cost, newFeat] = removefeat(X, Y, selectioncriteria, X_k)
%[n,d]=size(X);
N=length(X_k);

cost=zeros(N,N); % initialize costs table
for i=1:N 
    cost(i,2:N)=MysetDiff(X_k,X_k(i));
    subset=cost(i,2:end);
    cost(i,1)=eval(selectioncriteria);
end;

sorted_cost = sortrows(cost,1); % sort the costs
winner = sorted_cost(end,2:end); % remove the worst feature to winners
Cost = sorted_cost(end,1); % new cost after remove operation
newFeat=MysetDiff(X_k,sorted_cost(end,2:end)); % removed feat

% setA and setB are assumed to be two row vectors
function setC=MysetDiff(setA, setB)
na=length(setA);
nb=length(setB);
k=1;
for i=1:na
    eleman=setA(i);
    if(isempty(find(setB==eleman)))
        setC(k)=eleman;
        k=k+1;
    end;
end;%end i
return;