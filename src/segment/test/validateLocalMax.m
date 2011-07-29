clear all;
close all;

%[p, n, e] = fileparts(mfilename);


mask = imread('in-mask2-for-localmax.ppm');

lmax = imread('out-localmax.ppm') > 0;
lmin = imread('out-localmin.ppm') > 0;
lmax2 = imread('out-localmax2.ppm') > 0;
lmin2 = imread('out-localmin2.ppm') > 0;

tic; m_lmax = imregionalmax(mask, 8); t = toc;
disp(sprintf('matlab vs localmax %d took %f s', max(max(m_lmax ~= lmax)), t));
disp(sprintf('matlab vs localmax2 %d took %f s', max(max(m_lmax ~= lmax2)), t));

tic; m_lmin = imregionalmin(mask, 8); t=toc;
disp(sprintf('matlab vs localmin %d took %f s', max(max(m_lmin ~= lmin)), t));
disp(sprintf('matlab vs localmin2 %d took %f s', max(max(m_lmin ~= lmin2)), t));


hmin1 = imread('out-hmin-1.ppm');
hmin2 = imread('out-hmin-2.ppm');
hmin3 = imread('out-hmin-3.ppm');

tic; m_hmin1 = imhmin(mask, 1, 8); t=toc;
disp(sprintf('matlab vs hmin 1 %d took %f s', max(max(m_hmin1 ~= hmin1)), t));
tic; m_hmin2 = imhmin(mask, 2, 8); t=toc;
disp(sprintf('matlab vs hmin 2 %d took %f s', max(max(m_hmin2 ~= hmin2)), t));
tic; m_hmin3 = imhmin(mask, 3, 8); t=toc;
disp(sprintf('matlab vs hmin 3 %d took %f s', max(max(m_hmin3 ~= hmin3)), t));
