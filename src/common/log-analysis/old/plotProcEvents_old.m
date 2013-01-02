function [ img norm_events ] = plotProcEvents_old( proc_events, barWidth, pixelWidth, figname_prefix, allEventTypes, colorMap)
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
        end_bucket_start = (endt_bucket - 1) * pixelWidth + 1;
        start_bucket_start = (startt_bucket - 1) * pixelWidth + 1;
        end_bucket_end = endt_bucket * pixelWidth;
        
        % can't do this in more vectorized way, because of the range access
        for j = 1:length(types)
%           not faster...
%             % first mark the entire range
%             norm_events{i}(startt_bucket(j) : endt_bucket(j), typeIdx(j)) = ...
%                 norm_events{i}(startt_bucket(j) : endt_bucket(j), typeIdx(j)) + pixelWidth;
%             % then remove the leading
%             norm_events{i}(startt_bucket(j), typeIdx(j)) = ...
%                 norm_events{i}(startt_bucket(j), typeIdx(j)) - (startt(j) - start_bucket_start(j) + 1);
%             % then remove the trailing
%             norm_events{i}(endt_bucket(j), typeIdx(j)) = ...
%                 norm_events{i}(endt_bucket(j), typeIdx(j)) - (end_bucket_end(j) - endt(j) + 1);
            
            % if start and end in the same bucket, mark with duration
           if startt_bucket(j) == endt_bucket(j)
               norm_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   norm_events{i}(startt_bucket(j), typeIdx(j)) + endt(j) - startt(j) + 1;
           else
               % do the start first
               norm_events{i}(startt_bucket(j), typeIdx(j)) = ...
                   norm_events{i}(startt_bucket(j), typeIdx(j)) + start_bucket_end(j) - startt(j) + 1;
               % then do the end
               norm_events{i}(endt_bucket(j), typeIdx(j)) = ...
                    norm_events{i}(endt_bucket(j), typeIdx(j)) + endt(j) - end_bucket_start(j) + 1;
            
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
    sum_events = sparse(size(norm_events{1}, 1), size(norm_events{1}, 2));
	for i = 1:p	
		sum_events = sum_events + norm_events{i};
    end

    
    % allocate the image
    img = zeros(p, cols, 3, 'uint8');
    
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
        
        % convert from double to uint8
        bar2 = uint8(bar1 * 255.0);
        
        clear bar1;
        
        r0 = (i - 1) * (barWidth + 1) + 1;
        r1 = i * (barWidth + 1) - 1;
%         img(r0:r1, :, :) = ...
%             repmat(reshape(bar2, 1, size(bar2, 1), size(bar2, 2)), [barWidth, 1, 1]);
        img(i, :, :) = reshape(bar2, 1, size(bar2, 1), size(bar2, 2));
        clear bar2;
        clear r0;
        clear r1;
    end
    
    imwrite(img, [figname_prefix '.png'], 'png');

    sum_events = sparse(size(norm_events{1}, 1), size(norm_events{1}, 2));
	for i = 1:p	
		sum_events = sum_events + norm_events{i};
    end

    
    resolutionScaling = 3;
    % since we change the axes to reflect time values,
    fig = figure;
    
    sf1 = subplot(5,1,1:3);
    sa1 = gca;
    set(sa1, 'FontSize', get(sa1, 'FontSize') * resolutionScaling);

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
    
%    axes('position', [.1 .1 .8 .3], 'visible', 'on'); hold on;
    
    sf2 = subplot(5,1,4:5);
    sa2 = gca;
    set(sa2, 'FontSize', get(sa2, 'FontSize') * resolutionScaling);
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

    % remove the top graph's bottom axis,
    set(sa1,'XTickLabel','');
    % put the 2 graphs next to each other.
    pos1 = get(sa1, 'Position');
    pos2 = get(sa2, 'Position');
    pos2(4) = pos1(2) - pos2(2);
    set(sa2,'Position',pos2)

    % handle window resizing
    set(fig, 'ResizeFcn', {@rescaleAxes_old, sa2, pixelWidth});
    % limit panning and zooming.
    z = zoom;
    setAxesZoomMotion(z,sa2,'horizontal');
    p = pan;
    setAxesPanMotion(p,sa2,'horizontal');
    linkaxes([sa1 sa2], 'x');
    
    clear sum_events;
    
    % FOR PAPER only - eps files are big.
    %print(fig, '-depsc2', '-tiff', [figname_prefix '.fig.eps'], '-r300');
    %print(fig, '-dtiff', [figname_prefix '.fig.tif'], '-r1200');
    set(fig, 'InvertHardCopy', 'off', 'PaperPositionMode', 'auto');
    print(fig, '-dpng', [figname_prefix '.fig.png'], ['-r' num2str(100 * resolutionScaling)]);
    
end

function [] = rescaleAxes_old(src, eventdata, sa2, pixelW)
            
    xtic = get(sa2, 'XTick');
    timetic = xtic * pixelW / 1000000;
    set(sa2, 'XTickLabel', num2str(timetic'));

end
