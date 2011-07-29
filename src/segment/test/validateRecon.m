clear all;
close all;

%[p, n, e] = fileparts(mfilename);


mask = imread('in-mask.ppm');
marker = imread('in-marker.ppm');
maskb = imread('in-maskb.pbm') > 0;
markerb = imread('in-markerb.pbm') > 0;

recon = imread('out-recon.ppm');
reconb = imread('out-reconBin.pbm') > 0;

tic; m_recon = imreconstruct(marker, mask, 8); t= toc;
disp(sprintf('matlab vs recon %d took %f s', max(max(m_recon ~= recon)), t));

tic; m_reconb = imreconstruct(markerb, maskb, 8); t = toc;
disp(sprintf('matlab vs reconbin %d took %f s', max(max(m_reconb ~= reconb)), t));

bwao100_255 = imread('out-bwareaopen-100-255.pbm') > 0;
tic; m_bwao100_255 = bwareaopen(maskb, 100); t= toc;
disp(sprintf('matlab vs bwareaopen %d took %f s', max(max(m_bwao100_255 ~= bwao100_255)), t));

fid = fopen('out-bwlabel_4096_x_4096.raw', 'r');
bwlab = fread(fid, [4096,4096], 'int');
bwlab = bwlab';
fclose(fid);
tic; m_bwlab = bwlabel(maskb, 8); t= toc;
figure; imshow(label2rgb(bwlab));
figure; imshow(label2rgb(m_bwlab));

diff = (m_bwlab> 0) ~= (bwlab > 0);
disp(sprintf('matlab vs bwlabel %d took %f s', max(max(diff)), t));


