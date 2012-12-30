function [dataPerInterval si unique_vals] = resampleData( events, fields, sample_interval, allEventTypes, groupByField, timeInterval)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %% check the parameters.
    dataPerInterval = {};
    si = 0;
    unique_vals = {};
    
    if (size(events, 1) == 0)
        fprintf(2, 'WARN: no entries in input event log list\n');
        return;
    end
    
    if (size(events, 2) == 0)
        fprintf(2, 'WARN: no field in input event log list\n');
        return;
    end

    if (length(timeInterval) < 2)
        fprintf(2, 'ERROR: time interval needs a start and an end\n');
        return;
    end
    
    mn = double(timeInterval(1));
    mx = double(timeInterval(2));
    
    fns = fieldnames(fields);

    % now check the sampling interval
    si = sample_interval;
    if (isempty(sample_interval))
       si = min((mx- mn) / 50000.0, 1000000.0);
    end
    if (si > 1000000)
        fprintf(2, 'ERROR: sample interval should not be greater than 1000000 microseconds\n');
        return;
    end

    % only allow 'eventType' and 'eventName' as input types.
    if (isempty(find(strcmp(groupByField, {'eventType', 'eventName'}), 1)))
        fprintf(2, 'ERROR: can only compute throughput for eventType or eventName.  specified %s\n', groupByField);
        return;
    end
    
    if (isempty(find(strcmp('attribute', fns),1)))
        fprintf(2, 'WARN: no attributes.  cannot compute throughput\n');
        return;
    end
    
    
    %% first get some "constants"
    % MATLAB:  unique on dataset and maybe dataset concat is slow.  operate
    % on originam event_names first.
    p = size(events, 1);
    % get the unique event names, type mapping, and also the count
    if (strcmp(groupByField, 'eventName'))
        unique_vals = unique(cat(1, events{:, fields.(groupByField)}));
    else
        unique_vals = allEventTypes;
    end
    num_ev_vals = size(unique_vals, 1);
     
    cols = ceil((mx - mn) / double(si))+1;
    
    %% now compute event stats by name

    dataPerInterval = cell(p, 1);

        % p is number of cells (procs)
        % mx is maximum timestamp overall;


        % generate the sampled_events
        for i = 1:p

            dataPerInterval{i} = sparse(cols, num_ev_vals);

            vals = events{i, fields.(groupByField)};
            startt = double(events{i, fields.('startT')} - mn) ;
            endt = double(events{i, fields.('endT')} - mn) ;
            datasize = double(events{i, fields.('attribute')});

            [~, idx] = ismember(vals, unique_vals);
            clear names;

            startt_bucket = ceil(startt / si);
            endt_bucket = ceil(endt / si);

            start_bucket_end = startt_bucket * si;
            end_bucket_start = (endt_bucket - 1) * si + 1;

            duration = endt - startt + 1;  % data rate per microsec.

            %datarate = datasize ./ duration;
            fulldr = (datasize * si) ./ duration;
            % can't do this in more vectorized way, because of the range access

            startdr = datasize .* (start_bucket_end - startt + 1) ./ duration;
            enddr = datasize .* (endt - end_bucket_start + 1) ./ duration;

            clear startt;
            clear endt;

            clear start_bucket_end;
            clear end_bucket_start;

            clear duration;

            %tmpdata = zeros(cols, num_ev_names);
            for j = 1:length(datasize)
                x1 = startt_bucket(j);
                x2 = endt_bucket(j);
                y = idx(j);

               % if start and end in the same bucket, mark with duration
               if x1 == x2
%                    tmpdata(x1, y) = ...
%                        tmpdata(x1, y) + datasize(j);
                   dataPerInterval{i}(x1, y) = ...
                       dataPerInterval{i}(x1, y) + datasize(j);
               else
                   % do the start first
%                    tmpdata(x1, y) = ...
%                        tmpdata(x1, y) + startdr(j);
%                     % then do the end
%                    tmpdata(x2,y) = ...
%                        tmpdata(x2, y) + enddr(j);
                   dataPerInterval{i}(x1, y) = ...
                       dataPerInterval{i}(x1, y) + startdr(j);
                    % then do the end
                   dataPerInterval{i}(x2,y) = ...
                       dataPerInterval{i}(x2, y) + enddr(j);

                   % then do in between
                   if x2 > (x1 + 1)
%                         tmpdata(x1+1 : x2-1, y) = ...
%                            tmpdata(x1+1 : x2-1, y) + fulldr(j);
                        dataPerInterval{i}(x1+1 : x2-1, y) = ...
                           dataPerInterval{i}(x1+1 : x2-1, y) + fulldr(j);
                   end
               end 
            end
            clear x1;
            clear x2;
            clear y;
            clear startdr;
            clear enddr;
            clear fulldr;
            clear startt_bucket;
            clear endt_bucket;
            clear idx;
            clear datasize;


            %ndata_sizes{i} = sparse(tmpdata);
            %clear tmpdata;
        end
    
        
        

    clear num_ev_vals;
    clear p;
    clear cols;
    clear mx;
    clear mn;
end


