% INPUTS
% X: each row is a data point
% Y: label column vector
% new_size:  the final number of features to be kept
% start_size:the initial number of features

% OUTPUTS
% winner: the feature indices of the final selected features
% Cost:   the value of the objective function with winner
% winners:feature indices groups at each feature number step
% Costs:  the values of the objective function with winners

function [winner, Cost, winners, Costs] = SFFS(X, Y, new_size, start_size, selectioncriteria)
    if(start_size >= new_size)
       error('Start size cannot be equal to or greater than end size');
    end
    Costs=zeros(size(X,2),1); % Initialise, record subset costs
    Subset=zeros(size(X,2),size(X,2)); 
    % Start by running forward selection algorithm to form k=2
    % Function returns forward search vector of required size k
    [X_k, Costs(1:start_size)] = SFS(X, Y, start_size, selectioncriteria);
    for i=1:start_size
        Subset(i,1:i)=X_k(1:i);
    end;
    k = start_size;
while(k ~= new_size)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step-1 Inclusion
        % ----------------
        % Search through excluded features to find feature that combined with X_k 
        % produces best cost value Reinclude this value into X_k to make X_k+1.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [Subset(k+1,1:k+1), Costs(k+1), k_p_one] = addfeat(X, Y, selectioncriteria, Subset(k,1:k));% Find optimum cost plus one feat
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step-2 Test
        % -----------
        % 1. Search through features of X_k+1 to find the feature that has the least 
        %    effect (X_r) when removed from X_k+1
        % 2. If r=k+1, change k=k+1 goto Step-1.
        % 3. If r~=k+1 AND C(X_k+1 - X_r)<C(X_k)
        % 4. If k=2 (num features) put X_k = (X_k+1 - X_r) and C(X_k)=C(X_k+1 - X_r))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 1.
        [X_r, Cost_m_X_r, r] = removefeat(X, Y, selectioncriteria, Subset(k+1,1:k+1));
        % 2.
        %if(r == k_p_one) %k_p_one)
        if(Costs(k) >= Cost_m_X_r)   
            k=k+1;
            %goto step-1
            step3 = 0;
        % 3.  
        %elseif((r ~= k_p_one) & (Cost_m_X_r < Costs(k)))
        else % (Costs(k) < Cost_m_X_r)   
            %goto step-3
            step3 = 1;
        end
        % Check to see if we have reached the desired number of features? 
        % ===============================================================
        if(k == new_size)
            %fprintf('Done!');
            %Subset
            winner = Subset(k,1:k);
            Cost = Costs(k);   
            winners=Subset;
            Costs=Costs;
            break;
        end
        % No? Then Continue...
        % --------------------
        % 4.       
        if(k == start_size)    
            Subset(k,1:k) = X_r; 
            Costs(k) = Cost_m_X_r;
            Cost=Costs;
            %goto step-1            
        elseif(step3 == 1)   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Test-3 Exclusion
        % ----------------
        % 1. X_k' = X_k+1 - X_r (remove X_r)
        % 2. Search for least significant feature X_s
        % 3. If C(X_k' - X_s) < C(X_k-1) then X_k = X_k' and go to step 1,
        % no further backward search is performed.
        % 4. Put X_k-1'=X_k' and C(X_k)=C(X_k') and go to step 1.
        % 5. Goto step 3.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        step3=1;
        X_d = X_r; 
        while(step3 == 1)  
            %fprintf('Exclude\n');
            % 1. Remove X_r, already done :)
            % 2. Find least sig feat in new set
            [X_s, Cost_m_X_s, x_s] = removefeat(X, Y, selectioncriteria, X_d);% Find optimum cost minus one feat
            % 3.
            if(Cost_m_X_s <= Costs(k-1) )
                Subset(k,1:k) =  X_d; 
                Costs(k) = Cost_m_X_r;
                % Goto step-1
                step3=0; 
            % 4.   
            else %(Cost_m_X_s > Costs(k-1))
                X_d = X_s;
                k=k-1;    
                % Goto step-1
                % 5.  
                if(k == start_size)    
                    Subset(k,1:k) = X_d;
                    Costs(k) = Cost_m_X_r;
                    Cost=Costs;
                    % Goto step-1
                    step3=0;
                % 6.   
                else    
                    % Goto Step-3 
                    step3=1; 
                end  
            end
        end % end while
        end % End step-3 else goto step-1.
end    
return; %end main