function varargout = BarPlotBreak(varargin)

% BarBreakPlot(y,y_break_start,y_break_end,break_type,scale)
% Produces a plot who's y-axis skips to avoid unecessary blank space
% 
% INPUT
% 
% 'y_break_range'
% y_break_end
% break_type
%    if break_type='RPatch' the plot will look torn
%       in the broken space
%    if break_type='Patch' the plot will have a more
%       regular, zig-zag tear
%    if break_plot='Line' the plot will merely have
%       some hash marks on the y-axis to denote the
%       break
% scale = between 0-1, the % of max Y value that needs to subtracted from
% the max value bars
% USAGE:
% figure;
% BarPlotBreak([10,40,1000], 50, 60, 'RPatch', 0.85);
%
% Original Version developed by:
% Michael Robbins
% robbins@bloomberg.net
% michael.robbins@bloomberg.net
%
% Modified by: 
% Chintan Patel
% chintan.patel@dbmi.columbia.edu


% Modified by:
% Tony Pan
% tcpan@emory.edu
% WORK IN PROGRESS:
% TODO: check the subtractions to make sure +1 is not needed
% TODO: handle negative range and negative Y values:  shift Y, do the
% processing, then shift back.  mix of + and - in stack bar is going to be
% interesting to deal with.

% data
y_break_range = [NaN NaN];
break_type = 'Patch';
newvarargin = cell(0);
j = 1;
start = 1;
yid = 0;

if (nargin ==0)
    return;
end

if (ischar(varargin{1}) || (isscalar(varargin{1}) && ishandle(varargin{1})))
    % first could be 'v6' or AX
    start = 2;
    newvarargin{j} = varargin{1};
    j = j+1;
end

i = start;
stacked = 0;
% extract the relevent parameters and save the other ones.
while (i <= nargin)
   if (i == start && isnumeric(varargin{i}))
       if (nargin > start && isnumeric(varargin{i+1}))
           % more than 1 arg and next one is numeric.  so look at the next arg
           if (isscalar(varargin{i+1}))
               if (isscalar(varargin{i}))
                   % both first and second are scalars, so current is X,
                   % and next is Y (X,Y) or (X,Y,W)
                   newvarargin{j} = varargin{i};
                   j = j+1;
                   i = i+1;
               % else first is not scalar, so this is Y. (Y,W)
               end
           else
               % next is not scalar but numeric, and this is numeric, so
               % next is Y.  (X,Y) or (X,Y,W)
               newvarargin{j} = varargin{i};
               j = j+1;
               i = i+1;
           end
       % else only 1 numeric arg, so this is Y. (Y), (Y, ...)
       end

       yid = j
   % else it's not the first (and numeric) parameter, and it's not a
   % string, so just add it to the new list.

   elseif (ischar(varargin{i} ))
       if (strcmpi('break_range', varargin{i}))
           i = i+1;
           y_break_range = varargin{i};
           i = i+1;
           continue;
       elseif (strcmpi('break_type', varargin{i}))
           i = i+1;
           break_type = varargin{i};
           i = i+1;
           continue;
       elseif (strcmpi('stacked', varargin{i}))
           stacked = 1;
       % else it's some other string param.
       end
       
   end
   newvarargin{j} = varargin{i};
   j = j+1;
   i = i+1;
end

Y=newvarargin{yid};
if (stacked == 0 || size(Y, 2) == 1)
    maxY = max(reshape(Y, 1, numel(Y)));
    minY = min(reshape(Y, 1, numel(Y)));
else 
    maxY = max(sum(Y, 2));
    minY = min(sum(Y, 2));
end

if (length(find(isnan(y_break_range))) == 2 || ...
    maxY <= y_break_range(2))
    % invalid break range specified.  run the normal bar char
    [varargout{1:nargout}] = bar(newvarargin{:});
    return;
end
    
  
y_break_start = y_break_range(1);
y_break_end = y_break_range(2);

%y_break_mid   = (y_break_end-y_break_start)./2+y_break_start;
y_break = y_break_end-y_break_start;

if (stacked == 0 || size(Y, 2) == 1)
    Y(Y>=y_break_end
    Y(Y>=y_break_end)=Y(Y>=y_break_end)-y_break;
    newvarargin{yid} = Y;
else
    mark = Y(:, 1);
    prevmark = mark;
    for s = 1:size(Y,2)
        mark = prevmark + Y(:, s);
        % mark < break start:  dont change
        
        % mark is in break, but prev mark is below
        idx = mark > y_break_start && mark <= y_break_end && prevmark <= y_break_start;
        Y(idx, s) = y_break_start - prevmark(idx);
        % mark is in break, and prev mark is in break
        idx = mark > y_break_start && mark <= y_break_end && prevmark > y_break_start && prevmark <= y_break_end;
        Y(idx, s) = 0;
        
        % mark is above, but prev mark is below
        idx = mark > y_break_end && prevmark <= y_break_start;
        Y(idx, s) = Y(idx, s) - y_break;
        % mark is above, and prev mark is in break
        idx = mark > y_break_end && prevmark > y_break_start && prevmark <= y_break_end;
        Y(idx, s) = mark(idx) - y_break_end;
        % mark is above, and prev mark is above - no change

        prevmark = mark;
    end
    newvarargin{yid} = Y;
end
%find the max and min and cut max to 1.5 times the min
[varargout{1:nargout}] = bar(newvarargin{:});

xlim=get(gca,'xlim');
ytick=get(gca,'YTick');
[~,i]=min(ytick<=y_break_start);
y=(ytick(i)-ytick(i-1))./2+ytick(i-1);
dy=(ytick(2)-ytick(1))./10;
xtick=get(gca,'XTick');
x=xtick(1);

switch break_type
    case 'Patch',
		% this can be vectorized
        dx=(xlim(2)-xlim(1))./10;
        yy=repmat([y-2.*dy y-dy],1,6);
        xx=xlim(1)+dx.*[0:11];
		patch([xx(:);flipud(xx(:))], ...
            [yy(:);flipud(yy(:)-2.*dy)], ...
            [.8 .8 .8])
    case 'RPatch',
		% this can be vectorized
        dx=(xlim(2)-xlim(1))./100;
        yy=y+rand(101,1).*2.*dy;
        xx=xlim(1)+dx.*(0:100);
		patch([xx(:);flipud(xx(:))], ...
            [yy(:);flipud(yy(:)-2.*dy)], ...
            [.8 .8 .8])
    case 'Line',
        dx=(xtick(2)-xtick(1))./2;
		line([x-dx x   ],[y-2.*dy y-dy   ]);
		line([x    x+dx],[y+dy    y+2.*dy]);
		line([x-dx x   ],[y-3.*dy y-2.*dy]);
		line([x    x+dx],[y+2.*dy y+3.*dy]);
end;

%ytick(ytick>y_break_start)=ytick(ytick>y_break_start)+y_break_mid;

ytick(ytick>y_break_start)=ytick(ytick>y_break_start)+y_break;

for i=1:length(ytick)
   yticklabel{i}=sprintf('%d',ytick(i));
end;
set(gca,'yticklabel',yticklabel);


