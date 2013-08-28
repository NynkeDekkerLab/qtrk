
function graphs()

figure(1);
showcsvimg('u');

figure(2);
showcsvimg('dudr');

figure(3);
showcsvimg('dudz');

figure(5);
stdxz=dlmread('stdev-xz.txt');
plot(stdxz(:,2));


end

function d=showcsvimg(fn)
d=dlmread([fn '.txt']); imshow(normalize(d)); 
title(fn);
fprintf('%s: min=%f, max=%f\n', fn, min(d(:)), max(d(:)));
end