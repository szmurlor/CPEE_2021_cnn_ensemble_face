% % 
wyn=[]; wynVal=[];
Pred_y=[]; Pred_yVal=[]; 
ySVM1=[];ySVM2=[];ySVM3=[];ySVM4=[];ySVM5=[];
accSVM1=[];accSVM2=[];accSVM3=[];accSVM4=[];accSVM5=[];
errorSVM1=[];errorSVM2=[];errorSVM3=[];errorSVM4=[];errorSVM5=[];
svmModel1={};svmModel2={};svmModel3={};svmModel4={};svmModel5={};
theGammma1=[];theGammma2=[];theGammma3=[];theGammma4=[];theGammma5=[];
yTB1=[];yTB2=[];yTB3=[];yTB4=[];yTB5=[];
accTB1=[];accTB2=[];accTB3=[];accTB4=[];accTB5=[];
errorTB1=[];errorTB2=[];errorTB3=[];errorTB4=[];errorTB5=[];
theNTrees1=[];theNTrees2=[];theNTrees3=[];theNTrees4=[];theNTrees5=[];
liczba_cech=[];liczba_cech_swfit=[];liczba_cech_nca=[];liczba_cech_relieff=[];liczba_cech_mrmr=[];liczba_cech_chi2=[];

     %images = imageDatastore('E:\isic-archive.com',...
     %images = imageDatastore( 'C:\chmura\ALEX_obrazy_all',...
     % images = imageDatastore('C:\chmura\Obrazy_MUCT',...
     % images = imageDatastore( 'C:\chmura\ALEX_obrazy20',...
     % images = imageDatastore( 'C:\chmura\narzedzia',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_Monik1',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_ISIC',...
     %images = imageDatastore('C:\chmura\Obrazy_Melanoma_Monik1',...
     %images = imageDatastore( 'C:\chmura\ALEX_obrazy_all',...
     %images = imageDatastore( '/home/szmurlor/Nextcloud/BAZA_MUCT/baza_org/faces_MUCT',...
     %images = imageDatastore( 'C:\Users\szmurlor\Nextcloud\ALEX_obrazy_all',...
     images = imageDatastore( '/home/szmurlor/Nextcloud/ALEX_obrazy_all',...
     'IncludeSubfolders',true,...
    'ReadFcn', @customreader, ...
    'LabelSource','foldernames');
    

[uczImages,testImages] = splitEachLabel(images,0.7,'randomized');
for i=1:10
    i
%[trainingImages,validationImages] = splitEachLabel(uczImages,0.8,'randomized');
[trainingImages,validationImages] = splitEachLabel(uczImages,0.70+0.25*rand(1),'randomized');
numTrainImages = numel(trainingImages.Labels);

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
xu=double(featuresTrain);  maxu=max(abs(xu)); xu=xu./maxu;
xt=double(featuresTest);   xt=xt./maxu;
ducz=double(trainingImages.Labels);
dtest=double(testImages.Labels);
testLabels=testImages.Labels
liczba_cech = [liczba_cech size(xu,2)]

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
    'ValidationFrequency',numIterationsPerEpoch);
%'LearnRateSchedule','piecewise'...
%     'Plots','training-progress',...

%netTransfer = trainNetwork(trainingImages,layers,options);
netTransfer = trainNetwork(augimdsTrain,layers,options);
% wyswietl architekture modelu
% analyzeNetwork(netTransfer)
predictedLabels = classify(netTransfer,testImages);
%testLabels = testImages.Labels;
accuracy = (mean(predictedLabels == testLabels))*100;
%xu=double(featuresTrain);  maxu=max(abs(xu)); xu=xu./maxu;
%xt=double(featuresTest);   xt=xt./maxu;
%ducz=double(trainingImages.Labels);
%dtest=double(testLabels);
%liczba_cech = [liczba_cech size(xu,2)]
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

% Klasyfikatory SVM i TREEBAGGER
%STEPWISEFIT
[B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY]=stepwisefit([xu;xt],[ducz;dtest],'penter',.2);
aa=[find(INMODEL~=0)];
e=[];czulosc=[];precyzja=[];
xucz=xu(:,aa); 
xucz=xucz+0.03*randn(size(xucz)).*xucz;
xtest=xt(:,aa);
liczba_cech_swfit = [liczba_cech_swfit size(xucz,2)]


%%%TREEBAGGER
xucz=xu(:,aa); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,aa);

% piaty parametr true/false - czy badac, testowac rozne wartosci ntrees
[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, false, 600)
%[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, true)
yTB1 = [yTB1 yTB];
accTB1 = [accTB1 accTB];
errorTB1 = [errorTB1 errorTB];
%modelTB1 = [modelTB1 tbModel];
theNTrees1 = [theNTrees1 theNTrees];
Pred_y=[Pred_y  str2double(yTB)];
wyn=[wyn  accTB]; 


%NCA
nca=fscnca(xu, ducz)
%q2=find(nca.FeatureWeights>0.00000000001);%ISIC
q2=find(nca.FeatureWeights>0.0000001);%ISIC
%%%SVM%%%SVM
xucz=xu(:,q2); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q2);

liczba_cech_nca = [liczba_cech_nca size(xucz,2)]


%%TREEBAGGER
xucz=xu(:,q2);
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q2);

% piaty parametr true/false - czy badac, testowac rozne wartosci ntrees
[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, false, 600)
%[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, true)
yTB2 = [yTB2 yTB];
accTB2 = [accTB2 accTB];
errorTB2 = [errorTB2 errorTB];
%modelTB2 = [modelTB2 tbModel];
theNTrees2 = [theNTrees2 theNTrees];
Pred_y=[Pred_y  str2double(yTB)];
wyn=[wyn  accTB]; 


%%
%RELIEFF
[RANKED,WEIGHT] = relieff(xu,ducz,50);
q3=RANKED(1:2000);
%%%SVM
xucz=xu(:,q3); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q3);
liczba_cech_relieff = [liczba_cech_relieff size(xucz,2)]


%TREEBAGGER
xucz=xu(:,q3); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q3);
% piaty parametr true/false - czy badac, testowac rozne wartosci ntrees
[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, false, 600)
%[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, true)
yTB3 = [yTB3 yTB];
accTB3 = [accTB3 accTB];
errorTB3 = [errorTB3 errorTB];
%modelTB3 = [modelTB3 tbModel];
theNTrees3 = [theNTrees3 theNTrees];
Pred_y=[Pred_y  str2double(yTB)];
wyn=[wyn  accTB]; 

%%
%MRMR
[idx,scores] = fscmrmr(xu,ducz);
%bar(scores(idx))
q4=idx(1:2000);
xucz=xu(:,q4);
xtest=xt(:,q4);
liczba_cech_mrmr = [liczba_cech_mrmr size(xucz,2)]


%TREEBAGGER
xucz=xu(:,q4); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q4);
% piaty parametr true/false - czy badac, testowac rozne wartosci ntrees
[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, false, 600)
%[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, true)
yTB4 = [yTB4 yTB];
accTB4 = [accTB4 accTB];
errorTB4 = [errorTB4 errorTB];
%modelTB4 = [modelTB4 tbModel];
theNTrees4 = [theNTrees4 theNTrees];
Pred_y=[Pred_y  str2double(yTB)];
wyn=[wyn  accTB]; 

%%
%CHi2
[idx,scores] = fscchi2(xu,ducz);
 % bar(scores(idx))
q5=idx(1:2000);
xucz=xu(:,q5);
%xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q5);
liczba_cech_chi2 = [liczba_cech_chi2 size(xucz,2)]


%TREEBAGGER
xucz=xu(:,q5); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q5);
% piaty parametr true/false - czy badac, testowac rozne wartosci ntrees
[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, false, 600)
%[yTB, accTB, errorTB, tbModel, theNTrees] = do_treebagger(xucz,ducz, xtest, dtest, true)
yTB5 = [yTB5 yTB];
accTB5 = [accTB5 accTB];
errorTB5 = [errorTB5 errorTB];
%modelTB5 = [modelTB5 tbModel];
theNTrees5 = [theNTrees5 theNTrees];
Pred_y=[Pred_y  str2double(yTB)];
wyn=[wyn  accTB]; 

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


table_metrics = table(nclass,dokl_zesp,median_zesp,max_zesp,mean_zesp,odch_zesp)
