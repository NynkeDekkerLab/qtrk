function d=normalize(d)
minD=min(d(:));
maxD=max(d(:));
d= (d-minD)./(maxD-minD);
end