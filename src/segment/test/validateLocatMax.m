clear all;
close all;

%[p, n, e] = fileparts(mfilename);


mask = imread('in-mask2-for-localmax.ppm');

lmax = imread('out-localmax.ppm') > 0;
lmin = imread('out-localmin.ppm') > 0;
lmax2 = imread('out-localmax2.ppm') > 0;
lmin2 = imread('out-localmin2.ppm') > 0;

m_lmax = imregionalmax(mask, 8);
m_lmin = imregionalmin(mask, 8);
disp(sprintf('matlab vs localmax %d', max(max(m_lmax ~= lmax))));
disp(sprintf('matlab vs localmin %d', max(max(m_lmin ~= lmin))));
disp(sprintf('matlab vs localmax2 %d', max(max(m_lmax ~= lmax2))));
disp(sprintf('matlab vs localmin2 %d', max(max(m_lmin ~= lmin2))));


hmin1 = imread('out-hmin-1.ppm');
hmin2 = imread('out-hmin-2.ppm');
hmin3 = imread('out-hmin-3.ppm');

m_hmin1 = imhmin(mask, 1, 8) > 0;
m_hmin2 = imhmin(mask, 2, 8) > 0;
m_hmin3 = imhmin(mask, 3, 8) > 0;
disp(sprintf('matlab vs hmin 1 %d', max(max(m_hmin1 ~= hmin1))));
disp(sprintf('matlab vs hmin 2 %d', max(max(m_hmin2 ~= hmin2))));
disp(sprintf('matlab vs hmin 3 %d', max(max(m_hmin3 ~= hmin3))));
