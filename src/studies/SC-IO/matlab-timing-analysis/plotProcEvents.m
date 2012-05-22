function [ img norm_events ] = plotProcEvents( proc_events, barWidth, pixelWidth, figname_prefix)
%plotTiming draws an image that represents the different activities in MPI
%   The function first parses the data and generate a normalized event map
%   with dimension p x ((max t - min t)/pixelWidth) x eventTypes.  This has
%   values from 0 to 1.0, double, indicating the percent of time a
%   particular event type occupied a pixel's time duration.
%
%   we hardcode the available event types, and the color mapping (in HSL)
%   hue maps to type of event.  saturation maps to proportion of event
%   within a pixel.  a completely balanced node would have a color of gray.
%   value can be used for aggregating multiple nodes?
%
%   the norm_events matrix is then rendered as an image of dimension
%   ((p + 1) * barwidth) x ((max t - min t)/pixelWidth) x 3, unsigned char,
%   in RGB format
%   
%   Each horizontal bar is one process, each pixel is some time interval.
%   
    p = size(proc_events,1);

    % get global min and max and the unique events
    mn = inf;
    mx = 0;
    for i = 1:p 
	   mn = min([mn, min(proc_events{i, 6})]);
       mx = max([mx, max(proc_events{i, 7})]);
    end

    % hardcode the event types - total of 8 types
    allEventTypes = [-1 0 11 12 21 22 31 32 41 42 43 44 45 46]';
    colorMap = [0, 0, 0; ...                    % OTHER, -1, black
                120.0, 0.4, 1.0; ...      % COMPUTE, 0, green
                240.0, 0.4, 1.0; ...      % MEM_IO, 11, blue
                240.0, 0.4, 1.0; ...      % GPU_MEM_IO, 12, blue
                180.0, 0.4, 1.0; ...      % NETWORK_IO, 21, cyan
                300.0, 0.4, 1.0; ...      % NETWORK_WAIT, 22, magenta
                60.0, 0.4, 1.0; ...       % FILE_I, 31, yellow
                0.0, 0.4, 1.0; ...        % FILE_O, 32, red
                180.0, 1.0, 1.0; ...      % ADIOS_INIT, 41, cyan
                300.0, 1.0, 1.0; ...      % ADIOS_OPEN, 42, magenta
                240.0, 1.0, 1.0; ...      % ADIOS_ALLOC, 43, blue
                60.0, 1.0, 1.0; ...       % ADIOS_WRITE, 44, yellow
                0.0, 1.0, 1.0; ...        % ADIOS_CLOSE, 45, red
                120.0, 1.0, 1.0; ...      % ADIOS_FINALIZE, 46, green
	];
    colorMap(:, 1) = colorMap(:, 1) / 180.0 * pi;  % in radian
    % XY positions on colorwheel.
    colorMapCart = colorMap;
    [colorMapCart(:, 1) colorMapCart(:, 2)] = pol2cart(colorMap(:, 1), colorMap(:, 2));
    colorMapCart(find(abs(colorMapCart) < eps)) = 0;
    



    % get the number of pixels
    cols = ceil(double(mx) / double(pixelWidth));
    
    % allocate norm-events
	norm_events = cell(p, 1);   
% 	norm_events = zeros(p, cols, length(allEventTypes), 'double');
    
    % generate the norm_events
    for i = 1:p
        norm_events{i} = sparse(cols, length(allEventTypes));
        types = proc_events{i, 5};
        [blah typeIdx] = ismember(types, allEventTypes);
        clear blah;
        startt = double(proc_events{i, 6});
        endt = double(proc_events{i, 7});
        
        startt_bucket = ceil(double(startt) / double(pixelWidth));
        endt_bucket = ceil(double(endt) / double(pixelWidth));
        
        start_bucket_end = startt_bucket * pixelWidth;
        end_bucket_start = (endt_bucket - 1) * pixelWidth;
        
        % process the buckets
        for j = 1:length(types)
            % if start and end in the same bucket, mark with duration
           if startt_bucket(j) == endt_bucket(j)
               norm_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   norm_events{i}(startt_bucket(j), typeIdx(j)) + endt(j) - startt(j);
           else
               % do the start first
               norm_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   norm_events{i}(startt_bucket(j), typeIdx(j)) + start_bucket_end(j) - startt(j);
               % then do the end
               norm_events{i}(endt_bucket(j), typeIdx(j)) = ...
                    norm_events{i}(endt_bucket(j), typeIdx(j)) + endt(j) - end_bucket_start(j);
            
               % then do in between
               if endt_bucket(j) > (startt_bucket(j) + 1)
                    norm_events{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) = ...
                        norm_events{i}(startt_bucket(j)+1 : endt_bucket(j)-1, typeIdx(j)) + pixelWidth;
               end
           end 
        end
    	norm_events{i} = norm_events{i} / double(pixelWidth);
        
        clear types;
        clear startt;
        clear endt;
        clear startt_bucket;
        clear endt_bucket;
        clear start_bucket_end;
        clear end_bucket_start;
        
    end
    % normalize
    %norm_events = norm_events / double(pixelWidth);
    
    % allocate the image
    img = zeros(p * (barWidth + 1), cols, 3, 'double');
    
    % convert norm_events to hsv img
    for i = 1:p
        % get weighted sum of colors
        bar1 = norm_events{i} * colorMapCart;
        
        % convert from cartesian to polar coord (radians )
        [bar1(:, 1) bar1(:, 2)] = cart2pol(bar1(:, 1), bar1(:,2));
        
        % angle is between -pi and pi. (function uses atan2)
        bar1(:, 1) = bar1(:, 1) / (2.0 * pi);  % convert to -1/2 to 1/2
        idx = find(bar1(:,1) < 0);
        bar1(idx, 1) = bar1(idx, 1) + 1.0;  % convert to 0 to 1, via a "mod"
        bar1(find(abs(bar1(:,1)) < eps), 1) = 0;
        bar1(find(abs(bar1(:,1) - 1.0) < eps), 1) = 1;
        
        % convert hsvimg to rgb
        bar1 = hsv2rgb(bar1);
        bar1(find(abs(bar1) < eps)) = 0;
        bar1(find(abs(bar1 - 1.0) < eps)) = 1;
        
        
        r0 = (i - 1) * (barWidth + 1) + 1;
        r1 = i * (barWidth + 1) - 1;
        img(r0:r1, :, :) = ...
            repmat(reshape(bar1, 1, size(bar1, 1), size(bar1, 2)), [barWidth, 1, 1]);
        clear bar1;
        clear r0;
        clear r1;
    end
    
    imwrite(img, [figname_prefix '.tif'], 'tiff');
    
    fig = figure;
    subplot(2,1,1);
    imshow(img);
    axis on;
    axis normal;
    title('process activities: (BLACK:unknown) (GREEN:COMPUTE,ADIOS_FINALIZE) (BLUE:MEM_IO,ADIOS_ALLOC); RED:(FILE_O,ADIOS_CLOSE) (CYAN:NET_IO,ADIOS_INIT) (MAGENTA:NET_WAIT,ADIOS_OPEN) (YELLOW:FILE_I,ADIOS_WRITE)');

    sum_events = sparse(size(norm_events{1}, 1), size(norm_events{1}, 2));
	for i = 1:p	
		sum_events = sum_events + norm_events{i};
	end
    subplot(2,1,2);
    plot(sum_events(:, 1), '--k'); hold on;
    plot(sum_events(:, 2), '-.g'); hold on;
    plot(sum_events(:, 3), ':b'); hold on;
    plot(sum_events(:, 4), ':b'); hold on;
    plot(sum_events(:, 5), '-.c'); hold on;
    plot(sum_events(:, 6), ':m'); hold on;
    plot(sum_events(:, 7), ':y'); hold on;
    plot(sum_events(:, 8), '-.r'); hold on;
    plot(sum_events(:, 9), '-c'); hold on;
    plot(sum_events(:, 10), '-m'); hold on;
    plot(sum_events(:, 11), '-b'); hold on;
    plot(sum_events(:, 12), '-y'); hold on;
    plot(sum_events(:, 13), '-r'); hold on;
    plot(sum_events(:, 14), '-g'); hold on;
    axis tight;
    clear sum_events;
    
    print(fig, '-dtiff', [figname_prefix '.fig.tif']);
end

