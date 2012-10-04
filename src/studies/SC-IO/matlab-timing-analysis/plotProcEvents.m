function [ img norm_events sum_events ] = plotProcEvents( proc_events, barWidth, pixelWidth, figname_prefix, allEventTypes, colorMap)
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

    %% check the parameters.

    img = [];
    norm_events = [];
    
    if (isempty(proc_events))
        fprintf(2, 'ERROR: no events to process\n');
        return;
    end

    p = size(proc_events,1);

    % get global min and max and the unique events
    mx = 0;
    mn = inf;
    for i = 1:p 
	   mx = max([mx, max(proc_events{i, 7}, [], 1)], [], 2);
       mn = min([mn, min(proc_events{i, 6}, [], 1)], [], 2);
    end
    mn = mn - 1;  % do min as just out of range.

    % now check the sampling interval
    if (isempty(pixelWidth))
       pixelWidth = min((mx-mn ) / 50000, 1000000);
    end
    if (pixelWidth > 1000000)
        printf(2, 'ERROR: sample interval should not be greater than 1000000 microseconds\n');
        return;
    end

    % XY positions on colorwheel.
    colorMapCart = colorMap;
    [colorMapCart(:, 1) colorMapCart(:, 2)] = pol2cart(colorMap(:, 1), colorMap(:, 2));
    colorMapCart(abs(colorMapCart) < eps) = 0;
    

    % get the number of pixels
    cols = ceil(double(mx-mn) / pixelWidth) +1;
    clear mx;
    
    
    
    %% RESAMPLE
    % allocate norm-events
    num_ev_types = length(allEventTypes);
    sum_events = sparse(cols, num_ev_types);
    % allocate norm-events
    norm_events = cell(p, 1);

    % inline function shares variable so don't need to copy variables.
    function resampleTimeByType2
        % p is number of cells (procs)
        % mx is maximum timestamp overall;

        event_types = allEventTypes;
        sample_interval = pixelWidth;

        % generate the sampled_events
        for i = 1:p

            norm_events{i} = sparse(cols, num_ev_types);

            types = proc_events{i, 5};
            startt = double(proc_events{i, 6} - mn) ;
            endt = double(proc_events{i, 7} - mn) ;

            [~, idx] = ismember(types, event_types);
            clear types;

            startt_bucket = ceil(startt / sample_interval);
            endt_bucket = ceil(endt / sample_interval);

            start_bucket_end = startt_bucket * sample_interval;
            end_bucket_start = (endt_bucket - 1) * sample_interval + 1;

            duration = endt - startt + 1;  % data rate per microsec.

            % can't do this in more vectorized way, because of the range access
            startdr = start_bucket_end - startt + 1;
            enddr = endt - end_bucket_start + 1;
            clear startt;
            clear endt;
            clear start_bucket_end;
            clear end_bucket_start;

            tmp = zeros(cols, num_ev_types);
            for j = 1:length(duration)
                x1 = startt_bucket(j);
                x2 = endt_bucket(j);
                y = idx(j);

               % if start and end in the same bucket, mark with duration
               if x1 == x2
                   tmp(x1, y) = ...
                       tmp(x1, y) + duration(j);
               else
                   % do the start first
                   tmp(x1, y) = ...
                       tmp(x1, y) + startdr(j);
                    % then do the end
                   tmp(x2,y) = ...
                       tmp(x2, y) + enddr(j);

                   % then do in between
                   if x2 > (x1 + 1)
                        tmp(x1+1 : x2-1, y) = ...
                           tmp(x1+1 : x2-1, y) + sample_interval;
                   end
               end 
            end
            tmp = tmp / sample_interval;
            clear idx;
            clear startt_bucket;
            clear endt_bucket;
            clear duration;
            clear startdr;
            clear enddr;

            norm_events{i} = sparse(tmp);
            clear tmp;
        end

        for i = 1:p	
            sum_events = sum_events + norm_events{i};
        end
        clear num_ev_types;
    end
    
    %[norm_events sum_events] = resampleTimeByType(proc_events, p, cols, pixelWidth, allEventTypes);
    resampleTimeByType2;
    
    
    
    %% RENDER
    % allocate the image
    img = zeros(p, cols, 3, 'uint8');
    
    clear cols;
    
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
        bar1(abs(bar1(:,1)) < eps, 1) = 0;
        bar1(abs(bar1(:,1) - 1.0) < eps, 1) = 1;
        clear idx;
        
        % convert hsvimg to rgb
        bar1 = hsv2rgb(bar1);
        bar1(abs(bar1) < eps) = 0;
        bar1(abs(bar1 - 1.0) < eps) = 1;
        
        % convert from double to uint8
        bar2 = uint8(bar1 * 255.0);
        clear bar1;
        
%        r0 = (i - 1) * (barWidth + 1) + 1;
%        r1 = i * (barWidth + 1) - 1;
%         img(r0:r1, :, :) = ...
%             repmat(reshape(bar2, 1, size(bar2, 1), size(bar2, 2)), [barWidth, 1, 1]);
        img(i, :, :) = reshape(bar2, 1, size(bar2, 1), size(bar2, 2));
        clear bar2;
%        clear r0;
%        clear r1;
    end
    clear colorMapCart;
    
    % commented out - don't need these right now
    %imwrite(img, [figname_prefix '.png'], 'png');
    
    resolutionScaling = 3;
    % since we change the axes to reflect time values,
    fig = figure('visible', 'off');
    drawnow;
    
    sf1 = subplot(5,1,1:3);
    sa1 = gca;
    %set(sa1, 'FontSize', get(sa1, 'FontSize') * resolutionScaling);

    title(sa1, {'Process Activities',...
        '(BLACK:Unknown) (GREEN:Compute/ADIOS Finalize)', ...
        '(BLUE:Mem IO/ADIOS Alloc) (RED:File Write/ADIOS Close)',...
        '(CYAN:Net IO/ADIOS Init) (MAGENTA:Net Wait/ADIOS Open) (YELLOW:File Read/ADIOS Write)'});
    hold on;
%    axes('position', [.1 .4 .8 .5], 'visible', 'off'); hold on;
%    set(gca,'LooseInset',get(gca,'TightInset'));
    image(img); hold on;
    axis(sf1, 'tight');
%    xlabel(sa1, 'time (s)');
    ylabel(sa1, 'processes');
    clear sf1;

%    axes('position', [.1 .1 .8 .3], 'visible', 'on'); hold on;
    
    sf2 = subplot(5,1,4:5);
    sa2 = gca;
    %set(sa2, 'FontSize', get(sa2, 'FontSize') * resolutionScaling);
    plot(sa2, sum_events(:, 1), '--k'); hold on;
    plot(sa2, sum_events(:, 2), '-.g'); hold on;
    plot(sa2, sum_events(:, 3), ':b'); hold on;
    plot(sa2, sum_events(:, 4), ':b'); hold on;
    plot(sa2, sum_events(:, 5), '-.c'); hold on;
    plot(sa2, sum_events(:, 6), ':m'); hold on;
    plot(sa2, sum_events(:, 7), ':y'); hold on;
    plot(sa2, sum_events(:, 8), '-.r'); hold on;
    plot(sa2, sum_events(:, 9), '-c'); hold on;
    plot(sa2, sum_events(:, 10), '-m'); hold on;
    plot(sa2, sum_events(:, 11), '-b'); hold on;
    plot(sa2, sum_events(:, 12), '-y'); hold on;
    plot(sa2, sum_events(:, 13), '-r'); hold on;
    plot(sa2, sum_events(:, 14), '-g'); hold on;
    set(sa2, 'Color', [0.2 0.2 0.2]); hold on;  % some background color
    axis(sf2, 'tight');
    xlabel(sa2, 'time (s)');
    ylabel(sa2, 'num processes');
    clear sf2;

    % remove the top graph's bottom axis,
    set(sa1,'XTickLabel','');
    % put the 2 graphs next to each other.
    pos1 = get(sa1, 'Position');
    pos2 = get(sa2, 'Position');
    pos2(4) = pos1(2) - pos2(2);
    set(sa2,'Position',pos2)
    clear pos1;
    clear pos2;
        
    % handle window resizing
    set(fig, 'ResizeFcn', {@rescaleAxes, sa2, pixelWidth});
    % limit panning and zooming.
    z = zoom;
    setAxesZoomMotion(z,sa2,'horizontal');
    p = pan;
    setAxesPanMotion(p,sa2,'horizontal');
    linkaxes([sa1 sa2], 'x');
    clear z;
    clear p;
    
    % FOR PAPER only - eps files are big.
    %print(fig, '-depsc2', '-tiff', [figname_prefix '.fig.eps'], '-r300');
    %print(fig, '-dtiff', [figname_prefix '.fig.tif'], '-r1200');
    %set(fig, 'InvertHardCopy', 'off', 'PaperPositionMode', 'auto');
    set(fig, 'InvertHardCopy', 'off');
    set(fig, 'paperposition', [0 0 11 8.5]);
    set(fig, 'papersize', [11 8.5]);
    print(fig, '-dpng', [figname_prefix '.fig.png'], ['-r' num2str(100 * resolutionScaling)]);

    clear sa1;
    clear sa2;
    clear fig;
       
    clear p;
    clear map;

end

function [] = rescaleAxes(src, eventdata, sa2, pixelW)
            
    xtic = get(sa2, 'XTick');
    timetic = xtic * pixelW / 1000000;
    set(sa2, 'XTickLabel', num2str(timetic'));
    clear xtic;
    clear timetic;
end


function [ sampled_events sum_events ] = ...
    resampleTimeByType(events, p, cols, sample_interval, event_types, mn)
    % p is number of cells (procs)
    % mn is min timestamp, -1 timestamp overall;

    % get the number of pixels
    num_ev_types = length(event_types);
    
    % allocate norm-events
	sampled_events = cell(p, 1);
    
    % generate the sampled_events
    for i = 1:p

        sampled_events{i} = sparse(cols, num_ev_types);
        
        types = events{i, 5};
        startt = double(events{i, 6}- mn);
        endt = double(events{i, 7} - mn);
        
        [~, idx] = ismember(types, event_types);
        clear types;
        
        startt_bucket = ceil(startt / sample_interval);
        endt_bucket = ceil(endt / sample_interval);
        
        start_bucket_end = startt_bucket * sample_interval;
        end_bucket_start = (endt_bucket - 1) * sample_interval + 1;

        duration = endt - startt + 1;  % data rate per microsec.
        
        % can't do this in more vectorized way, because of the range access
        startdr = start_bucket_end - startt + 1;
        enddr = endt - end_bucket_start + 1;
        clear startt;
        clear endt;
        clear start_bucket_end;
        clear end_bucket_start;
        
        tmp = zeros(cols, num_ev_types);
        for j = 1:length(duration)
            x1 = startt_bucket(j);
            x2 = endt_bucket(j);
            y = idx(j);
            
           % if start and end in the same bucket, mark with duration
           if x1 == x2
               tmp(x1, y) = ...
                   tmp(x1, y) + duration(j);
           else
               % do the start first
               tmp(x1, y) = ...
                   tmp(x1, y) + startdr(j);
                % then do the end
               tmp(x2,y) = ...
                   tmp(x2, y) + enddr(j);

               % then do in between
               if x2 > (x1 + 1)
                    tmp(x1+1 : x2-1, y) = ...
                       tmp(x1+1 : x2-1, y) + sample_interval;
               end
           end 
        end
        tmp = tmp / sample_interval;
    	clear idx;
        clear startt_bucket;
        clear endt_bucket;
        clear duration;
        clear startdr;
        clear enddr;
        
        sampled_events{i} = sparse(tmp);
        clear tmp;
    end
    
    
    sum_events = sparse(cols, num_ev_types);
    for i = 1:p	
		sum_events = sum_events + sampled_events{i};
	end
    clear num_ev_types;
end

