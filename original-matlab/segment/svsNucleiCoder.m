%#codegen
function svsNucleiCoder(impath,filename,resultpath, folder,tile)
    
    % removed return var of "nuclei"
    if nargin==0
        image = {'astroII.1.ndpi-0000004096-0000004096.tif',...
            '2487_71614_48776_2462x1616.tif',...
            '2487_69236_37042_2462x1616.tif',...
            '2497_31102_51514_2462x1616.tif',...
            '2496_58448_61628_2462x1616.tif',...
            '2498_166520_32534_2462x1616.tif',...
            };
        for i =1
            impath='/Users/kongj/Glioma/Whole_TCGA_Slides/4K/code20x/';
            filename=image{i};
            resultpath=impath;
            svsNucleiCoder(impath,filename,resultpath,0,0);
        end
        return;
    end





    %============START================

    p.AN='NS-MORPH';    %TCGA dataset
    p.SN='1';
    p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=45;
    %p.SN='2';
    %p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=30;
    p.SR='20x';
    p.AR='20x';
    p.CR='1';


    % p.AN='NS-MORPH';    %VALIDATION dataset
    % p.SN='1';
    % p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=45;
    % %p.SN='2';
    % %p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=30;
    % p.SR='40x';      %Scanning Resolution   %40X--18slides; %20x--necrosis
    % p.AR='20x';      %Analysis Resolution
    % p.CR='0.5';

    %TONY - imread is not supported by matlab coder
    coder.extrinsic('imread', 'rgb2gray', 'tic', 'bwperim', 'imwrite', 'save');
    %END TONY
    
    %TONY
    I=zeros(4096,4096,3, 'uint8');
    assert(isa(I,'uint8'));
    %END TONY
    I=imread([impath,filename]);

    THR = 0.9;
    grayI = rgb2gray(I);
    area_bg = length(find(I(:,:,1)>=220&I(:,:,2)>=220&I(:,:,3)>=220));
    %TONY
    ratio = zeros(1, 1, 'double');
    %END TONY
    ratio = area_bg/numel(grayI);
    if ratio >= THR
        return;
    end

    %TONY
    [f,L] = segNucleiMorphMeanshift(I);
%    tic;
%    [f,L] = segNucleiMorphMeanshift(I);
%    t=toc;
    %END TONY


    % BW = L>0;
    % LL = zeros(size(BW));
    % boundaries = bwboundaries(BW);
    % num = length(boundaries);
    %
    % for i = 1:num
    %     b = boundaries{i};
    %
    %     if size(b,1) > 15
    %         b(:,2) = lowB(b(:,2));
    %         b(:,1) = lowB(b(:,1));
    %         if any(b(:)<0 | b(:)>4096)
    %           continue;
    %         end
    %         boundaries{i} = b;
    %     end
    %
    %     tempBW = roipoly(BW,b(:,2),b(:,1));
    %     if sum(tempBW(:)) == 0
    %         continue;
    %     end
    %
    %     LL(tempBW) = count;
    %     count = count + 1;
    % end

    %TONY
    assert(isa(L, 'double'));
    BW = false(size(L));
    %END TONY
    BW=bwperim(L>0,4);
    R=I(:,:,1); R(BW)=0;
    G=I(:,:,2); G(BW)=255;
    B=I(:,:,3); B(BW)=0;
    I=cat(3,R,G,B);
    imwrite(I, [resultpath, filename,'.grid4.jpg'], 'Quality',80);

    save([resultpath, filename,'.grid4.mat'],'f','L','t','-v7.3');

    %pais(resultpath,[filename '.grid4.mat'],resultpath,folder,tile,p);

end


%#codegen
function [f,L] = segNucleiMorphMeanshift(color_img)
    f = [];
    %TONY
    coder.extrinsic('bwselect', 'imopen', 'imreconstruct', 'imfill', 'bwlabel',...
        'regionprops', 'bwareaopen', 'imdilate', 'bwdist', 'imhmin', 'watershed',...
        'ismember');
    L = zeros(size(color_img, 1), size(color_img,2), 'double');
    %END TONY

    r = color_img( :, :, 1);
    g = color_img( :, :, 2);
    b = color_img( :, :, 3);

    %T1=2.5; T2=2;
    T1=5; T2=4;

    imR2G = double(r)./(double(g)+eps);
    bw1 = imR2G > T1;
    bw2 = imR2G > T2;
    ind = find(bw1);
    
    % TONY
    bw = false(size(color_img,1), size(color_img,2));
    rbc = false(size(imR2G));
    if ~isempty(ind)
        [rows, cols]=ind2sub(size(imR2G),ind);
        bw = bwselect(bw2, cols, rows, 8);
        rbc = bw & ((double(r)./(double(b)+eps)) > 1);
    end
%    if ~isempty(ind)
%        [rows, cols]=ind2sub(size(imR2G),ind);
%        rbc = bwselect(bw2,cols,rows,8) & (double(r)./(double(b)+eps)>1);
%    else
%        rbc = zeros(size(imR2G));
%    end
    %END TONY

    rc = 255 - r;
    % TONY
    se10 = [     0     0     0     0     1     1     1     1     1     1     1     1     1     1     1     0     0     0     0;...
     0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0;...
     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0;...
     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;...
     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0;...
     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0;...
     0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0;...
     0     0     0     0     1     1     1     1     1     1     1     1     1     1     1     0     0     0     0];
    rc_open = imopen(rc, se10);
    %    rc_open = imopen(rc, strel('disk',10));
    rc_recon = zeros(size(rc), 'uint8');
    % END TONY
    rc_recon = imreconstruct(rc_open,rc);
    diffIm = rc-rc_recon;

    G1=80; G2=45; % default settings
    %G1=80; G2=30;  % 2nd run

    bw1 = imfill(diffIm>G1,'holes');

    %CHANGE
    [L] = bwlabel(bw1, 8);
    %TONY
    stats = struct('Area', []);
    assert(isa(stats(1).Area, 'double'));
    %END TONY
    stats = regionprops(L, 'Area');
    areas = [stats.Area];
    
    %CHANGE
    ind = find(areas>10 & areas<1000);
    bw1 = ismember(L,ind);
    bw2 = diffIm>G2;
    ind = find(bw1);

    if isempty(ind)
        return;
    end

    [rows,cols] = ind2sub(size(diffIm),ind);
    % TONY
    bw = false(size(bw2));
    bw = bwselect(bw2,cols,rows,8);
    %seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc;
    seg_norbc = bw & ~rbc;
    seg_nohole = false(size(bw));
    seg_open = false(size(bw));
    se = [0 1 0; 1 1 1; 0 1 0];
    %END TONY
    seg_nohole = imfill(seg_norbc,'holes');
%    seg_open = imopen(seg_nohole,strel('disk',1));
    seg_open = imopen(seg_nohole,se);
    
    %CHANGE
    %TONY
    bwao = false(size(seg_open));
    bwao = bwareaopen(seg_open,30);
    seg_big = false(size(bwao));
    seg_big = imdilate(bwao, se);
    %    seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1));
    not_seg_big = false(size(seg_big));
    not_seg_big = ~seg_big;
    bwd = zeros(size(seg_big), 'double');
    bwd = bwdist(not_seg_big);
    distance = zeros(size(seg_big), 'double');
    distance = -bwd;
    distance(not_seg_big) = -Inf;
    %    distance = -bwdist(~seg_big);
    %distance(~seg_big) = -Inf;
    distance2 = zeros(size(distance), 'double');
    %END TONY

    distance2 = imhmin(distance, 1);

    %lines=ones(size(distance));
    %lines(watershed(distance2)==0)=0;

    %TONY clear distance %END TONY

    %TONY
    water = zeros(size(distance2), 'uint32');
    water = watershed(distance2);
%    seg_big(watershed(distance2)==0) = 0;
    seg_big(water==0) = 0;
    %END TONY
    seg_nonoverlap = seg_big;
    %seg_nonoverlap = lines & seg_big;

    %TONY clear lines distance2 %END TONY

    %CHANGE
    [L] = bwlabel(seg_nonoverlap, 4);
    stats = regionprops(L, 'Area');
    areas = [stats.Area];

    %CHANGE
    ind = find(areas>20 & areas<1000);

    if isempty(ind)
        return;
    end

    %CHANGE
    seg = ismember(L,ind);
    %[L, num] = bwlabel(seg,8);

    %CHANGE
    [L,num] = bwlabel(imfill(seg, 'holes'),4);

    % [Bounds] = bwboundaries(L>0,4);
    % Bounds = Bounds.';
    % figure;imshow(color_img,[]);impixelinfo; hold on;
    % for i=1:num
    %    bb=Bounds{i};
    %    plot(bb(:,2),bb(:,1));
    % end
    % keyboard;



    % BW = L>0;
    % L = zeros(size(BW));
    % boundaries = bwboundaries(BW);
    % num = length(boundaries);
    %
    % count = 1;
    % for i = 1:num
    %     b = boundaries{i};
    %
    %     if size(b,1) > 15
    %         b(:,2) = lowB(b(:,2));
    %         b(:,1) = lowB(b(:,1));
    %         if any(b(:)<0 | b(:)>4096)
    %           continue;
    %         end
    %         boundaries{i} = b;
    %     end
    %
    %     tempBW = roipoly(BW,b(:,2),b(:,1));
    %     if sum(tempBW(:)) == 0
    %         continue;
    %     end
    %
    %     L(tempBW) = count;
    %     count = count + 1;
    % end
    %
    % labels=unique(L(:));
    % count=0;
    % LL = zeros(size(BW));
    % for i = 1:length(labels)
    %     LL(L==labels(i))=count;
    %     count=count+1;
    % end
    %
    % L = LL;
    % num = max(L(:));
    % difflabels = setdiff(0:num,unique(L(:)));
    %
    % if ~isempty(difflabels)
    %     f=[];L=[];
    %     return;
    % end




    smallG = double(color_img(:,:,2))./(double(sum(color_img, 3))+eps);
    [Gx, Gy] = gradient(smallG);
    diffG = sqrt(Gx.*Gx+Gy.*Gy);
    BW_canny = edge(smallG,'canny');
    seSmall = strel('disk', 2);

    statsI = regionprops( L, smallG,...
        'Area','Perimeter',...
        'Eccentricity','MajorAxisLength','MinorAxisLength',...
        'Extent',...
        'MaxIntensity','MeanIntensity','MinIntensity',...
        'PixelIdxList');  %CHANGE

    fArea = cat(1,statsI.Area);
    fPerimeter = cat(1,statsI.Perimeter);
    fEccentricity = cat(1,statsI.Eccentricity);
    fCircularity = 4*pi * fArea./ (fPerimeter.^2);
    fMajorAxisLength = cat(1,statsI.MajorAxisLength);
    fMinorAxisLength = cat(1,statsI.MinorAxisLength);
    fExtent = cat(1,statsI.Extent);
    fMeanIntensity = cat(1,statsI.MeanIntensity);
    fMaxIntensity = cat(1,statsI.MaxIntensity);
    fMinIntensity = cat(1,statsI.MinIntensity);

    %indices = regionprops(L, 'PixelIdxList'); %get indices of objects once
    %props = regionprops(L, diffG, 'PixelValues'); %call regionprops once

    f=zeros(num,19);

    for i = 1:num
        %single_bw = L==i;

        %CHANGE
        pixOfInterest = smallG(statsI(i).PixelIdxList);
        %pixOfInterest = smallG(indices(i).PixelIdxList);

        fStdIntensity = std(pixOfInterest);
        [counts,binLocation] = imhist(im2uint8(pixOfInterest));
        prob = counts/sum(counts);
        fEntropy = entropy(prob);
        fEnergy = sum(prob.^2);
        fSkewness = skewness( double(pixOfInterest) );
        fKurtosis = kurtosis( double(pixOfInterest) );

        %eliminate FSD calculation
        %     ini = find( single_bw==1 );
        %     [ini_r ini_c] = ind2sub(size(single_bw),ini(1));
        %     Pts = bwtraceboundary( single_bw, [ini_r ini_c], 'NE' );
        %     complexPts = Pts(:,1) + j*Pts(:,2);
        %     S = abs(fft(complexPts))/size(complexPts,1);
        %     fFSD = S(3:end)/S(2);
        fFSD = zeros(5, 1);

        %---part II---

        %    stats = regionprops( bwlabel(single_bw, 8), diffG,...
        %        'PixelValues');
        %    pixOfInterest = stats.PixelValues;

        %CHANGE
        pixOfInterest = diffG(statsI(i).PixelIdxList);
        %pixOfInterest = diffG(indices(i).PixelIdxList);

        fMeanGradMag = mean(pixOfInterest);
        fStdGradMag = std(pixOfInterest);
        [counts,binLocation] = imhist(im2uint8(pixOfInterest));
        prob = counts/sum(counts);
        fEntropyGradMag = entropy(prob);
        fEnergyGradMag = sum(prob.^2);
        fSkewnessGradMag = skewness( double(pixOfInterest) );
        fKurtosisGradMag = kurtosis( double(pixOfInterest) );

        %---part III---
        %CHANGE
        bw_canny = BW_canny(statsI(i).PixelIdxList);
        %bw_canny = BW_canny(indices(i).PixelIdxList);
        %bw_canny = BW_canny(single_bw);

        fSumCanny = sum(bw_canny(:));

        %CHANGE
        fMeanCanny = fSumCanny / length(statsI(i).PixelIdxList);
        %fMeanCanny = fSumCanny / length(indices(i).PixelIdxList);
        %fMeanCanny = fSumCanny / sum(single_bw(:));

        %---default grade----
        fGrade = 0;

        %Aggregate features from part I, II, and III
        if length(fFSD) < 5
            fFSD = [fFSD(:)' zeros(1, 5-length(fFSD))];
        end

        featureVec = [,...
            fFSD(1), fFSD(2), fFSD(3), fFSD(4), fFSD(5),...
            fStdIntensity,...
            fEntropy, fEnergy, fSkewness, fKurtosis,...
            fMeanGradMag,...
            fStdGradMag,...
            fEntropyGradMag,fEnergyGradMag,fSkewnessGradMag,fKurtosisGradMag,...
            fSumCanny,fMeanCanny,...
            fGrade];

        f(i,:) = featureVec;
    end

    old_f = [fArea, fPerimeter,...
        fEccentricity, fCircularity,...
        f(:,1:5),...
        fMajorAxisLength, fMinorAxisLength, fExtent,...
        fMeanIntensity, fMaxIntensity, fMinIntensity,...
        f(:,6:end)];

    delta = 8;
    new_f = CytoplasmFeatures(L, color_img, smallG, delta);

    if size(old_f,1)~=size(new_f,1)
        fprintf('No. of rows in the old feature set is not equal to that of the new feature set!\n');
        Num_Row = min(size(old_f,1),size(new_f,1));
        f = [old_f(1:Num_Row,:) new_f(1:Num_Row,:)];
        return;
    end

    f = [old_f new_f];

end


function low_s = lowB(s)
    S = fft(s);

    order = 30;
    cutoff_freq = 30;
    c = fir1(order, cutoff_freq/length(s)/2);
    C = fft(c,length(s));
    low_s = ifft(C'.*S);
    low_s(end) = low_s(1);
end


function [features names] = CytoplasmFeatures(L, Color, Grayscale, delta)
    %Calculates cytoplasmic features by dilating nuclei segmentations to define
    %cytoplasm region for each nucleus.  Features are then calculated from the
    %hematoxlyin, eosin, and grayscale representations of the color image.
    %inputs:
    %L - label image (from bwlabel).
    %Color - color H&E image.
    %Grayscale - grayscale H&E image.
    %delta - radius of circular dilation kernel.  Recommended 8.
    %outputs:
    %feautres - N x ?? matrix of feature vectors in rows (N # of objects in L).
    %names - .

    [M N] = size(L); %tile size

    objects = max(L(:)); %get number of segmented nuclei in L

    %calculate stain intensities using color deconvolution
    stains =[0.650 0.072 0; 0.704 0.990 0; 0.286 0.105 0]; %H&E 2 color matrix
    Deconvolved = ColorDeconvolution(Color, stains, [true true false]);
    Hematoxylin = Deconvolved(:,:,1);
    Eosin = Deconvolved(:,:,2);
    clear Deconvolved;

    %dilate nuclei objects to capture surrounding cytoplasm
    disk = strel('disk', delta, 0); %create round structuring element
    bbox = regionprops(L, 'BoundingBox'); %calculate bounding box for each nucleus
    for i = 1:max(L(:)) %each nucleus must be processed independently to avoid collisions during dilation

        bounds = GetBounds(bbox(i).BoundingBox, delta, M, N); %get bounds of dilated nucleus

        BW = L(bounds(3):bounds(4), bounds(1):bounds(2)) == i; %grab field surrouding nucleus

        cytoplasm = xor(BW, imdilate(BW, disk)); %remove nucleus region from cytoplasm+nucleus mask

        PixelList{i} = PixIndex(cytoplasm, bounds, M, N); %get list of cytoplasm pixels

    end

    %calculate hematoxlyin features, capture feature names
    [HematoxylinIntensityGroup IntensityNames] = IntensityFeatureGroup(Hematoxylin, PixelList);
    [HematoxylinTextureGroup TextureNames] = TextureFeatureGroup(Hematoxylin, PixelList);
    [HematoxylinGradientGroup GradientNames] = GradientFeatureGroup(Hematoxylin, PixelList);

    %calculate eosin features
    EosinIntensityGroup = IntensityFeatureGroup(Eosin, PixelList);
    EosinTextureGroup = TextureFeatureGroup(Eosin, PixelList);
    EosinGradientGroup = GradientFeatureGroup(Eosin, PixelList);

    %calculate grayscale image features
    GrayscaleIntensityGroup = IntensityFeatureGroup(Grayscale, PixelList);
    GrayscaleTextureGroup = TextureFeatureGroup(Grayscale, PixelList);
    GrayscaleGradientGroup = GradientFeatureGroup(Grayscale, PixelList);

    %concatenate features
    features = [HematoxylinIntensityGroup HematoxylinTextureGroup HematoxylinGradientGroup...
        EosinIntensityGroup EosinTextureGroup EosinGradientGroup...
        GrayscaleIntensityGroup GrayscaleTextureGroup GrayscaleGradientGroup];

    %create names output
    featurenames = {IntensityNames{:} TextureNames{:} GradientNames{:}};
    names = cell(3 * (length(IntensityNames) + length(TextureNames) + length(GradientNames)), 1);
    %TONY added this to make it compatible with MATLAB CODER
    hem_prefix = cell(length(featurenames), 1);
    eos_prefix = cell(length(featurenames), 1);
    gray_prefix = cell(length(featurenames), 1);
    for i = 1:length(featurenames)
        hem_prefix{i} = 'Hematoxlyin';
        eos_prefix{i} = 'Eosin';
        gray_prefix{i} = 'Grayscale';
    end
    names(1:length(featurenames)) = strcat(hem_prefix, featurenames);
    names(length(featurenames)+1:2*length(featurenames)) = strcat(eos_prefix, featurenames);
    names(2*length(featurenames)+1:3*length(featurenames)) = strcat(gray_prefix, featurenames);
    %END TONY
%    names(1:length(featurenames)) = cellfun(@(x)strcat('Hematoxlyin', x), featurenames, 'UniformOutput', false);
%    names(length(featurenames)+1:2*length(featurenames)) = cellfun(@(x)strcat('Eosin', x), featurenames, 'UniformOutput', false);
%    names(2*length(featurenames)+1:3*length(featurenames)) = cellfun(@(x)strcat('Grayscale', x), featurenames, 'UniformOutput', false);

end

function [f names] = IntensityFeatureGroup(I, ObjectPixelList)
    %Calculate intensity feature group images from intensity image 'I' for
    %objects defined in ObjectPixelLists.
    %inputs:
    %I - intensity image.
    %ObjectPixelList - N-length cell array containing linear indices of N
    %                   objects in 'I'.
    %outputs:
    %f - N x 5 array containing features for object i in row i.
    %names - feature names.

    f = zeros(length(ObjectPixelList), 4); %initialize 'f'

    for i = 1:length(ObjectPixelList) %calculate for each object
        pixOfInterest = I(ObjectPixelList{i});
        f(i,1) = double(mean(pixOfInterest));
        f(i,2) = f(i,1) - double(median(pixOfInterest));
        f(i,3) = max(pixOfInterest);
        f(i,4) = min(pixOfInterest);
        f(i,5) = std(double(pixOfInterest));
    end

    names = {'MeanIntensity', 'MeanMedianDifferenceIntensity', 'MaxIntensity', 'MinIntensity', 'StdIntensity'};
end

function [f names]= TextureFeatureGroup(I, ObjectPixelList)
    %Calculate texture feature group images from intensity image 'I' for
    %objects defined in ObjectPixelLists.
    %inputs:
    %I - intensity image.
    %ObjectPixelList - N-length cell array containing linear indices of N
    %                   objects in 'I'.
    %outputs:
    %f - N x 4 array containing features for object i in row i.
    %names - feature names.

    f = zeros(length(ObjectPixelList), 4);
    for i = 1:length(ObjectPixelList)
        pixOfInterest = I(ObjectPixelList{i});
        [counts] = imhist(im2uint8(pixOfInterest));
        prob = counts/sum(counts);
        f(i,1) = entropy(prob);
        f(i,2) = sum(prob.^2);
        f(i,3) = skewness( double(pixOfInterest) );
        f(i,4) = kurtosis( double(pixOfInterest) );
    end

    names = {'Entropy', 'Energy', 'Skewness', 'Kurtosis'};

end

function [f names] = GradientFeatureGroup(I, ObjectPixelList)
    %Calculate gradient feature group images from intensity image 'I' for
    %objects defined in ObjectPixelLists.
    %inputs:
    %I - intensity image.
    %ObjectPixelLists - N-length cell array containing linear indices of N
    %                   objects in 'I'.
    %outputs:
    %f - N x 5 array containing features for object i in row i.
    %names - feature names.

    [Gx, Gy] = gradient(double(I));
    diffG = sqrt(Gx.*Gx+Gy.*Gy);
    BW_canny = edge(I,'canny');
    seSmall = strel('disk', 2);

    f = zeros(length(ObjectPixelList), 8);
    for i = 1:length(ObjectPixelList)
        pixOfInterest = diffG(ObjectPixelList{i});
        fMeanGradMag = mean(pixOfInterest);
        fStdGradMag = std(pixOfInterest);
        [counts,binLocation] = imhist(im2uint8(pixOfInterest));
        prob = counts/sum(counts);
        fEntropyGradMag = entropy(prob);
        fEnergyGradMag = sum(prob.^2);
        fSkewnessGradMag = skewness( double(pixOfInterest) );
        fKurtosisGradMag = kurtosis( double(pixOfInterest) );

        bw_canny = BW_canny(ObjectPixelList{i});
        fSumCanny = sum(bw_canny(:));

        fMeanCanny = fSumCanny / length(pixOfInterest);

        f(i,:) = [fMeanGradMag, fStdGradMag, fEntropyGradMag, fEnergyGradMag,...
            fSkewnessGradMag,fKurtosisGradMag, fSumCanny, fMeanCanny];
    end

    names = {'MeanGradMag', 'StdGradMag', 'EntropyGradMag', 'EnergyGradMag',...
        'SkewnessGradMag', 'KurtosisGradMag', 'SumCanny', 'MeanCanny'};
end

function bounds = GetBounds(bbox, delta, M, N)
    %get bounds of object in global label image
    bounds(1) = max(1,floor(bbox(1) - delta));
    bounds(2) = min(N, ceil(bbox(1) + bbox(3) + delta));
    bounds(3) = max(1,floor(bbox(2) - delta));
    bounds(4) = min(M, ceil(bbox(2) + bbox(4) + delta));
end

function idx = PixIndex(Binary, bounds, M, N)
    %get global linear indices of object extracted from tile
    [i j] = find(Binary);
    i = i + bounds(3) - 1;
    j = j + bounds(1) - 1;
    idx = sub2ind([M N], i, j);
end

%DEBUGGING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     nucleiedges{i} = cell2mat(bwboundaries(BW, 'noholes'));
%     nucleiedges{i}(:,1) = nucleiedges{i}(:,1) + bounds(3)-1;
%     nucleiedges{i}(:,2) = nucleiedges{i}(:,2) + bounds(1)-1;
%     nucleiedges{i} = sub2ind([M N], nucleiedges{i}(:,1), nucleiedges{i}(:,2));
%
%     cytoedges{i} = cell2mat(bwboundaries(cytoplasm, 'noholes'));
%     cytoedges{i}(:,1) = cytoedges{i}(:,1) + bounds(3)-1;
%     cytoedges{i}(:,2) = cytoedges{i}(:,2) + bounds(1)-1;
%     cytoedges{i} = sub2ind([M N], cytoedges{i}(:,1), cytoedges{i}(:,2));

%     [iL jL] = find(L == i);
%     [iBW jBW] = find(BW);
%     iBW = iBW + bounds(3) - 1;
%     jBW = jBW + bounds(1) - 1;
%     if(~isequal(iL,iBW) & ~isequal(jL,jBW))
%         error('Big problem ocurred.');
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%     nucleiedges = cell2mat(nucleiedges.');
%     cytoedges = cell2mat(cytoedges.');
%
%     R = Color(:,:,1);
%     G = Color(:,:,2);
%     B = Color(:,:,3);
%     R(nucleiedges) = 255;
%     G(nucleiedges) = 0;
%     B(nucleiedges) = 0;
%     R(cytoedges) = 0;
%     G(cytoedges) = 255;
%     B(cytoedges) = 0;
%     Color(:,:,1) = R;
%     Color(:,:,2) = G;
%     Color(:,:,3) = B;
%     figure; imshow(Color);


function intensity = ColorDeconvolution(I, M, stains)
    %Deconvolves multiple stains from RGB image I, given stain matrix 'M'.
    %inputs:
    %I - an MxNx3 RGB image in uint8 form range [0 - 255].
    %M - a 3x3 matrix containing the color vectors in columns. For two stain images
    %    the third column is zero.  Minumum two nonzero columns required.
    %stains - a logical 3-vector indicating which output channels to produce.
    %           Each element corresponds to the respective column of M.  Helps
    %           reduce memory footprint when only one output stain is required.
    %output:
    %intensity - an MxNxsum(stains) intensity image in uint8 form range [0 - 255].  Each
    %         channel is a stain intensity.  Channels are ordered same as
    %         columns of 'M'.

    %example inputs:
    % M =[0.650 0.072 0;  %H&E 2 color matrix
    %  0.704 0.990 0;
    %  0.286 0.105 0];
    %M = [0.29622945 0.47740915 0.72805583; %custom user H&E
    %     0.82739717 0.7787432 0.001;
    %    0.4771394  0.40698838 0.6855175];

    for i = 1:3 %normalize stains
        if(norm(M(:,i), 2))
            M(:,i) = M(:,i)/norm(M(:,i));
        end
    end

    if(norm(M(:,3), 2) == 0) %only two colors specified
        if ((M(1,1)^2 + M(1,2)^2) > 1)
            M(1,3) = 0;
        else
            M(1,3) = sqrt(1 - (M(1,1)^2 + M(1,2)^2));
        end

        if ((M(2,1)^2 + M(2,2)^2) > 1)
            M(2,3) = 0;
        else
            M(2,3) = sqrt(1 - (M(2,1)^2 + M(2,2)^2));
        end

        if ((M(3,1)^2 + M(3,2)^2) > 1)
            M(3,3) = 0;
        else
            M(3,3) = sqrt(1 - (M(3,1)^2 + M(3,2)^2));
        end

        M(:,3) = M(:,3)/norm(M(:,3));
    end

    Q = inv(M); %inversion
    Q = single(Q(logical(stains),:));

    dn=deconvolution_normalize(single(im2vec(I)));
    cn=Q * dn;
    channels = deconvolution_denormalize(cn);

    m = size(I,1); n = size(I,2);
    intensity = uint8(zeros(m, n, sum(stains)));

    for i = 1:sum(stains)
        intensity(:,:,i) = reshape(uint8(channels(i,:)), [m n]);
    end

end


function vec = im2vec(I)
    %converts color image to 3 x MN matrix

    M = size(I,1);
    N = size(I,2);

    if(size(I,3) == 3)
        vec = [I(1:M*N); I(M*N+1:2*M*N); I(2*M*N+1:3*M*N)];
    elseif(size(I,3) == 1)
        vec = [I(:)];
    else
        vec = [];
    end

    vec = double(vec);

end


function normalized = deconvolution_normalize(data)
    %Normalize raw color values according to Rufriok and Johnston's color
    %deconvolution scheme.
    %data - 3 x N matrix of raw color vectors (type double or single)
    %normalized - 3 x N matrix of normalized color vectors (type double or
    %single)

    normalized = -(255*log((data + 1)/255))/log(255);
end

function denormalized = deconvolution_denormalize(data)
    %de-normalize raw color values according to Rufriok and Johnston's color
    %deconvolution scheme.
    %data - 3 x N matrix of raw color vectors (type double or single)
    %denormalized - 3 x N matrix of normalized color vectors (type double or
    %single)

    denormalized = exp(-(data - 255)*log(255)/255);

end


