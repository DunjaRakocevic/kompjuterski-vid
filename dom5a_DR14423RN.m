clear; clc; close all;


dataFolder = "C:\Users\PC\Desktop\KV\5a i 5b\slikePrirode";
imds = imageDatastore(dataFolder,"IncludeSubfolders",true,"LabelSource","foldernames");



rng(1);
idx = randperm(numel(imds.Files),16);

figure;
tiledlayout(4,4,"Padding","compact","TileSpacing","compact");
for i=1:16
    nexttile;
    imshow(readimage(imds,idx(i)));
    title(string(imds.Labels(idx(i))));
end
saveas(gcf,"random_images.png");




[imdsTrain,imdsRest] = splitEachLabel(imds,0.8,"randomized");
[imdsVal,imdsTest]   = splitEachLabel(imdsRest,0.5,"randomized");


inputSize = [128 128 3];
numClasses = numel(categories(imds.Labels));

augTrain = augmentedImageDatastore(inputSize,imdsTrain);
augVal   = augmentedImageDatastore(inputSize,imdsVal);
augTest  = augmentedImageDatastore(inputSize,imdsTest);




layers = [

    imageInputLayer(inputSize)

    convolution2dLayer(3,12,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,24,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,48,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,"Stride",2)

    convolution2dLayer(3,96,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,"Stride",2)

    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer

];





figure;
plot(layerGraph(layers));
saveas(gcf,"cnn_architecture.png");


options = trainingOptions("adam", ...
    "InitialLearnRate",1e-3, ...
    "MaxEpochs",8, ...
    "MiniBatchSize",64, ...
    "Shuffle","every-epoch", ...
    "ValidationData",augVal, ...
    "ValidationFrequency",50, ...
    "Verbose",true, ...
    "Plots","training-progress");

[net,info] = trainNetwork(augTrain,layers,options);




figure;
plot(info.TrainingAccuracy,"LineWidth",2); hold on;
plot(info.ValidationAccuracy,"LineWidth",2);
grid on;
xlabel("Iteration");
ylabel("Accuracy (%)");
legend("Training","Validation");
title("Accuracy during training");
saveas(gcf,"accuracy_plot.png");



numParams = 0;
for i = 1:numel(net.Layers)
    L = net.Layers(i);
    if isprop(L,"Weights") && ~isempty(L.Weights)
        numParams = numParams + numel(L.Weights);
    end
    if isprop(L,"Bias") && ~isempty(L.Bias)
        numParams = numParams + numel(L.Bias);
    end
end
fprintf("Ukupan broj treniranih parametara: %d\n",numParams);



W = net.Layers(2).Weights;
W = (W - min(W,[],"all")) / (max(W,[],"all") - min(W,[],"all") + eps);


figure;
tiledlayout(3,4,"Padding","compact","TileSpacing","compact");
for k=1:size(W,4)
    nexttile;
    imshow(W(:,:,:,k));
end
saveas(gcf,"conv1_filters.png");




YPred = classify(net,augTest);
YTrue = imdsTest.Labels;


accTest = mean(YPred == YTrue);
fprintf("Test tačnost: %.2f%%\n",accTest*100);



figure;
confusionchart(YTrue,YPred);
title(sprintf("Accuracy = %.2f%%",accTest*100));
saveas(gcf,"confusion_matrix.png");




T = table(info.TrainingAccuracy',info.ValidationAccuracy', 'VariableNames',{'TrainingAccuracy','ValidationAccuracy'});
writetable(T,"accuracy_log.csv");
