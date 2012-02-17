function Mapping = StringMatch(A, B)

%create java hashtable
Hash = java.util.Hashtable;

%load items from 'B' into 'Hash'
for i = 1:length(B)
    
    %check if collision
    item = Hash.get(B{i});
    
    %insert 'B{i}' into hash
    if(isempty(item))
        Hash.put(B{i}, i);
    else
        Hash.put(B{i}, [item.' i]);
    end
       
end

%initialize output
Mapping = cell(1, length(A));

%Get mapping from 'A' items to 'B' items
for i = 1:length(A)
    
    %recover mapped indices
    Item = Hash.get(A{i});
    if(size(Item,1) > size(Item,2))
        Item = Item.';
    end
    Mapping{i} = Item;
    
end