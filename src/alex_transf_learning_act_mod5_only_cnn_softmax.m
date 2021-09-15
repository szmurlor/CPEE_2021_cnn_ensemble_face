% % 
wyn=[]; wynVal=[];
Pred_y=[]; Pred_yVal=[]; 

     %images = imageDatastore('E:\isic-archive.com',...
     %images = imageDatastore( 'C:\chmura\ALEX_obrazy_all',...
     % images = imageDatastore('C:\chmura\Obrazy_MUCT',...
     % images = imageDatastore( 'C:\chmura\ALEX_obrazy20',...
     % images = imageDatastore( 'C:\chmura\narzedzia',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_Monik1',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_ISIC',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_Monik1',...
     %images = imageDatastore( 'C:\chmura\ALEX_obrazy_all',...
     %images = imageDatastore( '/home/szmurlor/Nextcloud/ALEX_obrazy_all',...
     images = imageDatastore( '/home/szmurlor/Nextcloud/BAZA_MUCT/baza_org/faces_MUCT',...
     'IncludeSubfolders',true,...
    'ReadFcn', @customreader, ...
    'LabelSource','foldernames');
    
% testImages = imageDatastore('C:\chmura\Obrazy_Melanoma_Monik',... 
%     'IncludeSubfolders',true,...
%     'ReadFcn', @customreader, ...
%     'LabelSource','foldernames');

[uczImages,testImages] = splitEachLabel(images,0.7,'randomized');
for i=1:10
    i
[trainingImages,validationImages] = splitEachLabel(uczImages,0.8,'randomized');
[trainingImages,validationImages] = splitEachLabel(uczImages,0.70+0.25*rand(1),'randomized');
numTrainImages = numel(trainingImages.Labels);

% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(trainingImages,idx(i));
%     imshow(I)
% end

net = alexnet;
% net.Layers
% layersTransfer = net.Layers(1:end-3);
layersTransfer = net.Layers(1:end-6);
inputSize = net.Layers(1).InputSize;

imageAugmenter = imageDataAugmenter( ...
'RandRotation',[-20,20], ...
'RandXTranslation',[-3 3], ...
'RandYTranslation',[-3 3]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages,'DataAugmentation',imageAugmenter);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
%augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);
layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

numClasses = numel(categories(trainingImages.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(700+35*i,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    reluLayer()
    dropoutLayer(0.4+0.23*rand(1))   
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

miniBatchSize = 10; % bylo 10
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',20,...
    'ExecutionEnvironment','gpu',...
    'InitialLearnRate',1.3e-4,...
    'Verbose',true,...
    'ValidationPatience',10,...% by³o 10
    'ValidationData',validationImages,...
    'Plots','training-progress',...
    'ValidationFrequency',numIterationsPerEpoch);
%'LearnRateSchedule','piecewise'...
    %'Plots','training-progress',...

%netTransfer = trainNetwork(trainingImages,layers,options);
netTransfer = trainNetwork(augimdsTrain,layers,options);
% wyswietl architekture modelu
% analyzeNetwork(netTransfer)
predictedLabels = classify(netTransfer,testImages);
testLabels = testImages.Labels;
accuracy = (mean(predictedLabels == testLabels))*100;
xu=double(featuresTrain);  maxu=max(abs(xu)); xu=xu./maxu;
xt=double(featuresTest);   xt=xt./maxu;
ducz=double(trainingImages.Labels);
dtest=double(testLabels);
%  xu1=xu;
%  xu=[xu+0.03*randn(size(xu)).*xu
%      xu];
%  ducz=[ducz;ducz];
%  aaa=randperm(2000);
%  xu=xu(aaa',:);
%  ducz=ducz(aaa');
% xucz=double(featuresTrain);
% xtest=double(featuresTest);
% ducz=double(trainingImages.Labels);
% dtest=double(testLabels);
% wyn=[wyn accuracy]

Pred_y=[double(Pred_y) double(predictedLabels)];
wyn=[wyn accuracy]

end
nclass=numClasses

%Zespó³
dy=[dtest Pred_y]
nclass=numClasses
clear dz
[lw,lk]=size(dy);
for i=1:lw
    for j=1:nclass
        d=0;
      for k=2:lk
        if (j==double(dy(i,k)))
            d=d+1;
        else
        end
      end
      
       dz(i,j)=d;
%      klasa=dy(:,1)
    end
end
dz;
[q,cl_max]=max(dz');
zgod_class=length(find(double(dy(:,1))==cl_max'));
dokl_zesp=zgod_class/lw*100
median_zesp=median(wyn)
max_zesp=max(wyn)
mean_zesp=mean(wyn)
odch_zesp=std(wyn)


[C,order] = confusionmat(double(testLabels),double(cl_max));
Confusion=C

 
for i=1:nclass,
     sen(i)=C(i,i)/sum(C(i,:));
 end
% 
 for i=1:nclass,
     prec(i)=C(i,i)/sum(C(:,i));
 end
 
lc1t=sum(dtest==1);
lc2t=sum(dtest==2);
 %ROC
 for i=1:length(dtest)
     ydz(i)=dz(i,1)/(dz(i,1)+dz(i,2));
 end
 for i=1:lc1t
   dtestroc(i)=1;
 end
 for i=lc1t+1:length(dtest)
   dtestroc(i)=0;
 end
 sen
 prec
 F1_zesp=2*prec.*sen./(prec+sen) 
[tpr,fpr,th]=roc(dtestroc,ydz);
%figure(1), plot(fpr,tpr)
%grid, xlabel('FPR'),ylabel('TPR'), title ('ROC')
%[X,Y,T,AUC] = perfcurve(dtestroc,ydz,1);
%AUC

table(wyn')

metrics_names = {'nclass','dokl_zesp','median_zesp','max_zesp','mean_zesp','odch_zesp'}';
metrics_names = string(metrics_names)
metrics_values = [nclass,dokl_zesp,median_zesp,max_zesp,mean_zesp,odch_zesp]';
table_metrics = table(metrics_names, metrics_values)

