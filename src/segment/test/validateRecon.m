clear all;
close all;

%[p, n, e] = fileparts(mfilename);


mask = imread('in-mask.ppm');
marker = imread('in-marker.ppm');
recon = imread('out-recon.ppm');

tic; m_recon = imreconstruct(marker, mask, 8); t= toc;
fprintf(1, 'matlab vs recon %d took %f s\n', max(max(m_recon ~= recon)), t);

maskb = imread('in-maskb.pbm') > 0;
markerb = imread('in-markerb.pbm') > 0;
reconb = imread('out-reconBin.pbm') > 0;


tic; m_reconb = imreconstruct(markerb, maskb, 8); t = toc;
fprintf(1, 'matlab vs reconbin %d took %f s\n', max(max(m_reconb ~= reconb)), t);

bwaolarge = imread('out-bwareaopen-large.pbm') > 0;
tic; m_bwaolarge = bwareaopen(maskb, 500); t= toc;
fprintf(1, 'matlab vs bwareaopen %d took %f s\n', max(max(m_bwaolarge ~= bwaolarge)), t);

fid = fopen('out-bwlabel_4096_x_4096.raw', 'r');
bwlab = fread(fid, [4096,4096], 'int');
bwlab = bwlab';
fclose(fid);
tic; m_bwlab = bwlabel(maskb, 8); t= toc;
%figure; imshow(label2rgb(bwlab));
%figure; imshow(label2rgb(m_bwlab));

diff = (m_bwlab> 0) ~= (bwlab > 0);
fprintf(1, 'matlab vs bwlabel %d took %f s\n', max(max(diff)), t);


recon = imread('out-recon4.ppm');
tic; m_recon = imreconstruct(marker, mask, 4); t= toc;
fprintf(1, 'matlab vs recon4 %d took %f s\n', max(max(m_recon ~= recon)), t);

reconb = imread('out-reconBin4.pbm') > 0;
tic; m_reconb = imreconstruct(markerb, maskb, 4); t = toc;
fprintf(1, 'matlab vs reconbin4 %d took %f s\n', max(max(m_reconb ~= reconb)), t);



%imfilldata = imread('text.png') > 0;
%maskb = repmat(imfilldata, 16, 16);
maskb = rgb2gray(imread('sizePhantom.ppm')) > 0;

bwaolarge = imread('out-bwareaopen4-large.pbm') > 0;
tic; m_bwaolarge = bwareaopen(maskb, 500, 4); t= toc;
fprintf(1, 'matlab vs bwareaopen4 %d took %f s\n', max(max(m_bwaolarge ~= bwaolarge)), t);

[L] = bwlabel(maskb, 8);
stats = regionprops(L, 'Area');
areas = [stats.Area]

%CHANGE
ind = find(areas >= 500);
bw1 = ismember(L,ind);

fprintf(1, 'matlab bwlabel and area vs bwareaopen4 %d took %f s\n', max(max(m_bwaolarge ~= bw1)), t);
