imageinfo = h5read('bcrTCGA.features-summary.h5', '/image-info');
imagesum = h5read('bcrTCGA.features-summary.h5', '/image-sum');
globalmean = h5readatt('bcrTCGA.features-summary.h5', '/image-sum', 'mean');
globalstdev = h5readatt('bcrTCGA.features-summary.h5', '/image-sum', 'stdev');
count = h5readatt('bcrTCGA.features-summary.h5', '/image-sum', 'count');

% read the exclusion file
incl = true(size(imageinfo.img_name));
fid = fopen('TCGAExclusions.txt', 'r');
        while 1
            tline = fgetl(fid);
            if ~ischar(tline), break, end
            [i,j]=ind2sub(size(imageinfo.img_name), strmatch(tline, imageinfo.img_name));
            incl(i, j) = false;
        end
fclose(fid);

imgnames = imageinfo.img_name(incl);
imgcounts = imagesum.nu_count(incl);
imgsums = imagesum.nu_sum(:, incl);
imgsums2 = imagesum.nu_sum_square(:, incl);

ptnames = char(imgnames);
ptnames = ptnames(:, 1:12);
ptnames = unique(ptnames, 'rows');
ptnames = cellstr(ptnames);

ptcounts = zeros(size(ptnames,1), 1, 'uint64');
ptmeans = zeros(size(imgsums, 1), size(ptnames,1), 'double');
ptstdevs = zeros(size(imgsums2, 1), size(ptnames,1), 'double');


normmean = ptmeans;
normstdev = ptstdevs;

for k = 1 : size(ptnames, 1)
    [i,j]=ind2sub(size(imgnames), strmatch(ptnames(k), imgnames));
    ptcounts(k) = sum(imgcounts(i));
    ptmeans(:, k) = sum(imgsums(:, i), 2) ./ double(ptcounts(k));
    ptstdevs(:, k) = sqrt(sum(imgsums2(:, i), 2) ./ double(ptcounts(k)) - ptmeans(:, k) .* ptmeans(:, k) );

    normmean(:, k) = (ptmeans(:, k) - globalmean) ./ globalstdev;
    normstdev(:, k) = ptstdevs(:, k) ./ globalstdev;
end


