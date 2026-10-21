close all
clear all

pkg load image


im = imread('slike\testpattern1024.tif');
im = im2double(im);



function imf = f_nelin(im, ksize)



  [rw, col] = size(im);

  r1 = floor((ksize-1)/2);
  r2 = ksize - r1 - 1;


  imP= zeros(rw + r1 + r2, col + r1 + r2);
  imP(r1+1:r1+rw, r1+1:r1+col) = im;

  imf = zeros(rw, col);

  for i = 1:rw
    for j = 1:col

      region = imP(i:i+ksize-1, j:j+ksize-1);

      imf(i,j) = max(region(:));

    end
  end

end


imf = f_nelin(im, 5);
figure, imshow(imf,[]), title('Moja funkcija sa 5');


imf2 = f_nelin(im, 4);
figure, imshow(imf2, []), title('Moja funkcija sa 4');





pom = ones(5);
imfbydilate = imdilate(im, pom);
figure, imshow(imfbydilate,[]), title('imdilate funkcija sa 5');


diff = abs(imf - imfbydilate);
figure, imshow(diff);
title('Apsolutna razlika izmedju rezultata funkcija');

fprintf('Maksimalna razlika: %e\n', max(diff(:)));
fprintf('Srednja razlika: %e\n', mean(diff(:)));

