clear all;
close all;

%[p, n, e] = fileparts(mfilename);


imfilldata = imread('imfillTest.pbm');
im = repmat(imfilldata, 512, 512);
seedsdata = false(size(imfilldata));
seedsdata(3,3) = 1;
seeds = repmat(seedsdata, 512, 512);

filled = imread('out-imfilled.pbm') > 0;
tic; m_filled = imfill(im, find(seeds), 8); t=toc;
disp(sprintf('matlab vs imfill %d took %f s', max(max(m_filled ~= filled)), t));

filledholes = imread('out-holesfilled.pbm') > 0;
tic; m_filledholes = imfill(im, 8, 'holes'); t=toc;
disp(sprintf('matlab vs imfill holes %d took %f s', max(max(m_filledholes ~= filledholes)), t));
    

imfilldata = imread('tire.tif');
im = repmat(imfilldata, 20, 17);

filledholes_g = imread('out-holesfilled-gray.ppm');
tic; m_filledholes_g = imfill(im, 8, 'holes'); t=toc;
disp(sprintf('matlab vs imfill holes gray %d took %f s', max(max(m_filledholes_g ~= filledholes_g)), t));


imfilldata = imread('text.png');
im = repmat(imfilldata, 16, 16);
seedsdata = false(size(imfilldata));
seedsdata(126,34) = 1;
seedsdata(187,172) = 1;
seedsdata(11,20) = 1;
seeds = repmat(seedsdata, 16,16);


bws = imread('out-bwselected.pbm')>0;
[r, c] = find(seeds);
tic; m_bws = bwselect(im, c, r, 8); t=toc;
disp(sprintf('matlab vs bwselect %d took %f s', max(max(m_bws ~= bws)), t));
