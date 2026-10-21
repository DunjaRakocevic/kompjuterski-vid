close all
clear all

pkg load image


cim = imread('slike\cat.jpg');
cim = im2double(cim);

dim = imread('slike\dog.jpg');
dim = im2double(dim);



figure, imshow(cim,[])
figure, imshow(dim,[])


[crw  ccol cch] = size(cim)
[drw  dcol dch] = size(dim)

if crw!=drw ||  ccol!=dcol || cch!=dch
   error("Pictures are of different dimentions");
end



function imOut = lowpass (im,sd)

  imOut = zeros(size(im));



  for ch = 1:size(im,3)

    chan = im (:,:,ch);

    [rw, col] = size(chan);

    [x, y] = meshgrid(0:col-1, 0:rw-1);


    u0 = floor(col/2);
    v0 = floor(rw/2);


    H = exp( -((x - u0).^2 + (y - v0) .^2)/(2*sd^2) );
    F = fftshift(fft2(chan));

    G = H .* F;




    imOut(:,:,ch) = real(ifft2(ifftshift(G)));

  end

end


dLow = lowpass(dim,16);


cLow = lowpass(cim,18);
cHigh = cim - cLow;





figure, imshow(dLow,[])
figure, imshow(cHigh,[])



hybrid = dLow + cHigh;


figure, imshow(hybrid,[])


