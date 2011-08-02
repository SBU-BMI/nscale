clear all;
close all;

% create a circle

z = false(4096,4096);

rmax = 1024;
center = rmax;
r = rmax-1;
d = 2*r+1;
i = -r : r;
i = repmat(i, d, 1);
j = i';

start = 1;
   
while r >= 0
   minid = center-r;
   maxid = center+r;
   
   regioni = i(minid:maxid, minid:maxid);
   regionj = j(minid:maxid, minid:maxid);
   
   ii = reshape(regioni, d*d, 1);
   jj = reshape(regionj, d*d, 1);
   
   dist = ii.*ii + jj.*jj;
   w = dist <= (r * r);
   w2 = reshape(w, d, d);
   z(start:start+d-1, start:start+d-1) = w2;
   imshow(z);
   
   start = start + d+1;  
   r = ((r+1) / 2) - 1 ;
   d = 2*r+1;

end

