function SegmentationProcessTiles(Slide, Desired, T, Folder)
%Uses openslide library to process tiles in slide.
%Paths must be set to OpenSlide, ColorDeconvolution and BoundaryValidator.
%inputs:
%Slide - string containing slide path and filename with extension.
%Desired - scalar, desired magnification.
%T - scalar, tilesize, natural number.
%Folder - path to outputs

%parameters
M = [-0.154 0.035 0.549 -45.718; -0.057 -0.817 1.170 -49.887]; %color normalization - discriminate tissue/background.
TargetImage = '/home/tahsin//GBM_target1.tiff';

%add paths
addpath /home/tahsin/work/matlab/OpenSlide/
addpath /home/tahsin/work/matlab/ColorDeconvolution/
addpath /home/tahsin/work/matlab/ColorNormalization/
addpath /home/tahsin/work/matlab/BoundaryValidator/

%check if slide can be opened
Valid = openslide_can_open(Slide);

%slide is a valid file
if(Valid)
    
    %get slide dimensions, zoom levels, and objective information
    [Dims, Factors, Objective, MppX, MppY] = openslide_check_levels(Slide);
    
    %get objective magnification of level 1
    Objective = str2double(Objective);
    
    %calculate magnification level
    if(~isnan(Objective))
        Magnifications = Objective ./ Factors;
        [~, Level] = min(abs(Magnifications - Desired));
    else
        Level = 1;
    end
    
    %calculate normalization parameters
    [tMean tStd] = TargetParameters(imread(TargetImage), M);
    
    
    %work through tiles, horizontally first, then vertically
    for i = 0 : T : Dims(Level, 1)
        for j = 0: T : Dims(Level, 2)
            
            %start timer
            tic;
            
            %generate coordinates for filename
            X = sprintf('%06.0f', j);
            Y = sprintf('%06.0f', i);
            
            %update console
            fprintf('\tProcessing tile %s.%s, %d of %d ', X, Y,...
                1 + (j/T) + (i/T)*ceil(Dims(Level,2)/T),...
                ceil(Dims(Level,1)/T) * ceil(Dims(Level,2)/T));            
            
            %read in tile
            I = openslide_read_regions(Slide, Level-1, j, i, T, T);
            
            %normalize color
            RGB = ColorNormalization_Tile(I{1}, tMean, tStd, M);
            
            %foreground/background segmentation
            Foreground = JunForeground(RGB);
            
            %proceed if nuclei are present
            if(sum(Foreground(:)) > 0)
                
                %individual cell segmentation
                [Label, bX, bY] = JunIndividual(Foreground);
                
                %if labeled objects exist
                if(~isempty(bX))
                    
                    %feature extraction
                    [Features, Names, cX, cY] = JunFeatureExtraction(Label, RGB);
                    
                    %continue processing if objects were located
                    if(~isempty(Features))
                        
                        %place boundaries, centroids in global frame
                        cX = cX+j;
                        cY= cY+i;
                        bX = cellfun(@(x) x+j, bX, 'UniformOutput', false);
                        bY = cellfun(@(x) x+i, bY, 'UniformOutput', false);
                        
                        %embed boundaries and output tile
                        Mask = bwperim(Label > 0, 4);
                        R = RGB(:,:,1); R(Mask) = 0;
                        G = RGB(:,:,2); G(Mask) = 255;
                        B = RGB(:,:,3); B(Mask) = 0;
                        I = cat(3,R,G,B);
                        
                        %locate slide file extension
                        Slashes = strfind(Slide, '/');
                        Dots = strfind(Slide, '.');
                        
                        %write results to disk
                        save([Folder Slide(Slashes(end)+1:Dots(end)-1) '.' X '.' Y '.mat'],...
                            'Foreground', 'Label', 'bX', 'bY', 'Features', 'Names', 'cX', 'cY');
                        imwrite(I, [Folder Slide(Slashes(end)+1:Dots(end)-1) '.' X '.' Y '.jpg']);
                        
                        %merge colinear points on boundaries
                        for k = 1:length(bX)
                            [bX{k}, bY{k}] = MergeColinear(bX{k}, bY{k});
                        end
                        
                        %generate database txt file
                        SegmentationReport([Folder Slide(Slashes(end)+1:Dots(end)-1) '.seg.' X '.' Y '.txt'],...
                            Slide(Slashes(end)+1:Dots(end)-1), cX, cY, Features, Names, bX, bY);
                        
                    end
                end
            end
            
            %update console
            fprintf('%g seconds.\n', toc);
            
        end
    end
    
else
    
    %display error
    error(['Cannot open slide ' Slide]);
    
end
