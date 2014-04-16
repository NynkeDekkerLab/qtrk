function d=normalize(d)
d=double(d);
minD=min(d(:));
maxD=max(d(:));
d= (d-minD)./(maxD-minD);
end