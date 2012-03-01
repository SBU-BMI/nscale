fid = fopen('out-bwareaopen-test--111.32SC1.raw', 'r');
d2 = fread(fid, [4096,4096], '*int32');
d2 = uint32(d2');
fclose(fid);
fid = fopen('out-bwareaopen-test--151.32SC1.raw', 'r');
d3 = fread(fid, [4096,4096], '*int32');
d3 = uint32(d3');
fclose(fid);
fid = fopen('out-bwareaopen-test--101.32SC1.raw', 'r');
d1 = fread(fid, [4096,4096], '*int32');
d1 = uint32(d1');
fclose(fid);
[count1, count2, cooccur] = compareLabels(d2, d3);
unique(sum(cooccur, 1))
unique(sum(cooccur, 2))
[count1, count2, cooccur] = compareLabels(d1, d2);
unique(sum(cooccur, 1))
unique(sum(cooccur, 2))
[count1, count2, cooccur] = compareLabels(d1, d3);
unique(sum(cooccur, 2))
unique(sum(cooccur, 1))