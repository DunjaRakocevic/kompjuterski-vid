close all
clear all

pkg load image


A = im2double(imread('slike/apple.jpeg'));
B = im2double(imread('slike/orange.jpeg'));



figure, imshow(A,[]);
figure, imshow(B,[]);




nd=3;
levels = nd+1;


[hA,wA,~] = size(A);
[hB,wB,~] = size(B);

h = min(hA,hB);
w = min(wA,wB);

A = imresize(A, [h w]);
B = imresize(B, [h w]);



lvl = 2^nd;
rw = floor(h/lvl)*lvl;
col = floor(w/lvl)*lvl;

if rw !=h || col!=w

  A = imresize(A, [rw col]);
  B = imresize(B, [rw col]);


  h=rw;
  w=col;

end



function g= gauss1D(sd)

  if sd < 0.5
    g=[1 1]/2;
    return;
  endif

  r = ceil(3*sd);
  x = -r: r;

  g = exp(-(x.^2)/(2*sd^2));
  g = g / sum(g);
  g = g(:)';

end


sigmafor1  = 0.01 * min(h,w);
sigmafor5 = 0.05 * min(h,w);

gfor1 = gauss1D(sigmafor1);
gfor5 = gauss1D(sigmafor5);








function G = gaussPyr(im, gaussFilt, lvl)


  G = cell(1, lvl);

  r = floor((length(gaussFilt)-1)/2);
  H = gaussFilt' * gaussFilt;

  chan=size(im,3);

  G{1} = im;
  curr = im;

  for l = 2:lvl


    currPad = padarray(curr, [r r], 'replicate');

    imOut = zeros(size(curr));

    for ch = 1:chan
      tmp = conv2(currPad(:,:,ch), H, 'same');
      imOut(:,:,ch) = tmp(1+r:end-r, 1+r:end-r);
    end

    curr = imOut(1:2:end, 1:2:end, :);

    G{l} = curr;


  end

end


function expanded = expand(level, gaussFilt)


  [r, c, ch] = size(level);


  h1d = 2 * gaussFilt;
  H = h1d' * h1d;


  expanded = zeros(2*r, 2*c, ch);
  expanded(1:2:end, 1:2:end, :) = level;



  pad = floor((length(h1d)-1)/2);
  expandedPad = padarray(expanded, [pad pad], 'replicate');


  for i = 1:ch
    tmp = conv2(expandedPad(:,:,i), H, 'same');
    expanded(:,:,i) = tmp(1+pad:end-pad, 1+pad:end-pad);
  end
end






function L = laplacPyr(im, gaussFilt, lvl)




  G = gaussPyr(im, gaussFilt, lvl);
  L = cell(1, lvl);


  for l = 1:lvl-1


    nextLvl = G{l+1};
    nextExp = expand(nextLvl,gaussFilt);


    L{l} = G{l} - nextExp;


  end


  L{lvl} = G{lvl};


end


GAtemp = gaussPyr(A,gfor1,levels);
GBtemp = gaussPyr(B,gfor1,levels);

LA = laplacPyr(A, gfor1, levels);
LB = laplacPyr(B, gfor1, levels);

for l = 1:levels
  subplot(1, levels, l);

  imshow(LB{l} + 0.5);
  title(sprintf('Level %d', l));
end

#{

for l = 1:levels

  subplot(1, levels, l);
  imshow(GBtemp{l} );
  title(sprintf('Level %d', l));
end

#}


MA=zeros(size(A));
MA(:,1:floor(end/2),:) = 1;

MB= 1-MA;


#figure, imshow(MA,[]);


function maskOut = blurMask(mask, gaussFilt)

  r =floor((length(gaussFilt)-1)/2);
  H = gaussFilt' * gaussFilt;

  chan=size(mask,3);


  maskOut = zeros(size(mask));


  pad = r;
  maskPad = padarray(mask, [pad pad], 'symmetric');

  for ch = 1:chan
    tmp = conv2(maskPad(:,:,ch), H, 'same');
    maskOut(:,:,ch) = tmp(1+pad:end-pad, 1+pad:end-pad);
  end

end


MA = blurMask(MA,gfor5);
MB = blurMask(MB,gfor5);

GMA = gaussPyr(MA, gfor5, levels);
GMB = gaussPyr(MB, gfor5, levels);


#{

for l = 1:levels
  subplot(1, levels, l);
  imshow(GMB{l});
  title(sprintf('Level %d', l));
end



for l = 1:levels
  subplot(1, levels, l);
  imshow(GMB{l})
  title(sprintf('Level %d', l));
end
#}



LAB = cell(1,levels);


for l= 1:levels
  LAB{l}= LA{l} .* GMA{l} + LB{l} .* GMB{l};
end





function imRecon = reconstruction(LAB,gaussFilt)



  lvl = length(LAB);

  imRecon = LAB{lvl};


  for l= lvl-1:-1:1

    imRecon = expand(imRecon, gaussFilt);
    imRecon = imRecon + LAB{l};


  end

  imRecon = min(max(imRecon, 0), 1);
end







AB = reconstruction(LAB,gfor1);





#figure, imshow(MA,[]);
#figure, imshow(MB,[]);

figure, imshow(AB,[]);

figure;

for l = 1:levels
  subplot(1, levels, l);
  imshow(LAB{l} + 0.5);
  title(sprintf('Level %d', l));
end







#TESTIRANJE


## TEST 1: Rekonstrukcija A
Arecons = reconstruction(LA, gfor1);
diff = abs(A - Arecons);
figure, imshow(diff, []), title('Apsolutna razlika |A - Arecons|');


fprintf('\n[TEST 1: Rekonstrukcija A]\n');
fprintf('Maksimalna razlika: %e\n', max(diff(:)));
fprintf('Srednja razlika: %e\n', mean(diff(:)));



##TEST 2: Relacija nivoa

G2recon = LA{2} + expand(GAtemp{2+1}, gfor1);
diff = abs(GAtemp{2} - G2recon);
figure, imshow(diff, []), title('Razlika |G2 - (L2 + expand(G3))|');


fprintf('\n[TEST 2: Relacija nivoa]\n');
fprintf('Maksimalna razlika: %e\n', max(diff(:)));
fprintf('Srednja razlika: %e\n', mean(diff(:)));



##TEST 3: Rekonstrukcija nivoa

function Gtarget = reconstructToLvl(L, gaussFilt, targetLevel)

  lvl = length(L);
  Gtarget = L{lvl};

  for l = lvl-1:-1:targetLevel
    Gtarget = expand(Gtarget, gaussFilt);
    Gtarget = Gtarget + L{l};
  end
end


G2fromL = reconstructToLvl(LA, gfor1, 2);
diff = abs(GAtemp{2} - G2fromL);
figure, imshow(diff, []), title('Razlika |G2 - rekonstruisani G2 iz LA|');


fprintf('\n[TEST 3: Rekonstrukcija G2]\n');
fprintf('Maksimalna razlika: %e\n', max(diff(:)));
fprintf('Srednja razlika: %e\n', mean(diff(:)));

