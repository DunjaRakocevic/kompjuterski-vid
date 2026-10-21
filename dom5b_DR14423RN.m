clear; clc; close all;


dataFolder = "C:\Users\PC\Desktop\KV\5a i 5b\slikePrirode";
imds = imageDatastore(dataFolder,"IncludeSubfolders",true,"LabelSource","foldernames");


[imdsTrain, imdsRest] = splitEachLabel(imds,0.8,"randomized");
[imdsVal, imdsTest]   = splitEachLabel(imdsRest,0.5,"randomized");




netAlex = alexnet;

inputSize = netAlex.Layers(1).InputSize(1:2);
numClasses = numel(categories(imds.Labels));


augTrain = augmentedImageDatastore(inputSize,imdsTrain);
augVal   = augmentedImageDatastore(inputSize,imdsVal);
augTest  = augmentedImageDatastore(inputSize,imdsTest);




layersTransfer = netAlex.Layers;
layersTransfer(end-2) = fullyConnectedLayer(numClasses,"Name","fc8_new");
layersTransfer(end)   = classificationLayer("Name","cls_new");

lgraph = layerGraph(layersTransfer);

figure;
plot(lgraph);
saveas(gcf,"alexnet_architecture.png");



options = trainingOptions("adam", ...
    "InitialLearnRate",1e-4, ...
    "MaxEpochs",6, ...
    "MiniBatchSize",64, ...
    "Shuffle","every-epoch", ...
    "ValidationData",augVal, ...
    "ValidationFrequency",50, ...
    "Verbose",true, ...
    "Plots","training-progress");


[net,info] = trainNetwork(augTrain,lgraph,options);




figure;
plot(info.TrainingAccuracy,"LineWidth",2); hold on;
plot(info.ValidationAccuracy,"LineWidth",2);
grid on;
xlabel("Iteration");
ylabel("Accuracy (%)");
legend("Training","Validation");
title("Accuracy during training (AlexNet)");
saveas(gcf,"alexnet_accuracy_plot.png");


W = net.Layers(2).Weights;
W = (W - min(W,[],"all")) / (max(W,[],"all") - min(W,[],"all") + eps);




figure;
tiledlayout(4,4,"Padding","compact","TileSpacing","compact");
for k=1:16
    nexttile;
    imshow(W(:,:,:,k));
end
saveas(gcf,"alexnet_conv1_filters.png");

YPred = classify(net,augTest);
YTrue = imdsTest.Labels;


accTest = mean(YPred == YTrue);
fprintf("AlexNet test tačnost: %.2f%%\n",accTest*100);



figure;
confusionchart(YTrue,YPred);
title(sprintf("AlexNet Accuracy = %.2f%%",accTest*100));
saveas(gcf,"alexnet_confusion_matrix.png");
