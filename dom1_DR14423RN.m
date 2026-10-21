close all
clear all

pkg load image


im = imread('slike\testpattern1024.tif');
im = im2double(im);


[rw  col] = size(im);


sd = 0.005* max(rw,col);

r=ceil(3*sd);



[x, y] = meshgrid(-r:r, -r:r);


K= 1/(2*pi*sd^2);
filt = (1/K) * exp( -(x .^2 + y .^2)/(2*sd^2) );
filt = filt / sum(filt(:));


figure, imshow(filt,[]);




function imf = f_lin(im, filt)

  [rw, col] = size(im);
  [m , n] = size(filt);


  r = floor(m/2);


  imP = zeros(rw+2*r, col+2*r);

  imP(r+1:r+rw, r+1:r+col) = im;



  imf = zeros(rw, col);

  for i = 1:rw
    for j = 1:col

      region = imP(i:i+2*r, j:j+2*r);

      imf(i,j) = sum(sum(region .* filt));

    end
  end


end



imf = f_lin (im, filt);
figure, imshow(imf,[]), title('Moja funkcija');

imfiltered = imfilter(im, filt, 0);
figure, imshow(imf,[]), title('imfilter funkcija');




diff = abs(imf - imfiltered);
figure, imshow(diff);
title('Apsolutna razlika izmedju rezultata funkcija');

fprintf('Maksimalna razlika: %e\n', max(diff(:)));
fprintf('Srednja razlika: %e\n', mean(diff(:)));
