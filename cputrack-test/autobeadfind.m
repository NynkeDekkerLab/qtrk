function autobeadfind(image, smp)
    if nargin==0
        image=normalize(imread('00008153.jpg'));
        %smp = imread('00008153-s.jpg');
        smp = imread('00008153-nc.jpg'); % badly centered
        smp=normalize(smp(:,:,1));
    end

    image = image-mean(image(:));
    image = makepowerof2(image);
    s = size(image);
    
    h = size(smp);
    
    tmp = zeros(s);
    corner = int32(s/2-h/2);
    tmp (corner(1):corner(1)+h(1)-1, corner(2):corner(2)+h(2)-1) = smp-mean(smp(:));%smp(end:-1:1,end:-1:1)-mean(smp(:));
 %   tmp = tmp(end:-1:1,end:-1:1);
    %imshow(normalize(fftshift(tmp)));
    conv = ifft2(fft2(fftshift(tmp)).*fft2(image));
    
    %imshow( [tmp normalize(abs(conv)); image image+normalize(abs(conv)) ] );
    imshow(image+normalize(abs(conv))*2);
end

function d=normalize(d)
    d=double(d);
    minD = min(d(:));
    maxD = max(d(:));
    
    d=(d-minD)/(maxD-minD);
end

function resized = makepowerof2(img)
    s=size(img);
    p=nextpow2(s);
    %p=[max(p) max(p)];
    resized=zeros(2.^p);
    resized(1:s(1),1:s(2))=img;
end

