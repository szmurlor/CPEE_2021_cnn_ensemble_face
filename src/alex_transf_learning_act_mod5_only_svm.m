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
     %images = imageDatastore( '/home/szmurlor/Nextcloud/ALEX_obrazy_all',...
     images = imageDatastore( 'C:\Users\szmurlor\Nextcloud\ALEX_obrazy_all',...
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
%netTransfer = trainNetwork(augimdsTrain,layers,options);
% wyswietl architekture modelu
%analyzeNetwork(netTransfer)
%predictedLabels = classify(netTransfer,testImages);
%accuracy = (mean(predictedLabels == testLabels))*100;
%Pred_y=[double(Pred_y) double(predictedLabels)];
%wyn=[wyn accuracy]

% Klasyfikatory SVM i TREEBAGGER
%STEPWISEFIT
[B,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY]=stepwisefit([xu;xt],[ducz;dtest],'penter',.2);
aa=[find(INMODEL~=0)];
e=[];czulosc=[];precyzja=[];
xucz=xu(:,aa); 
xucz=xucz+0.03*randn(size(xucz)).*xucz;
xtest=xt(:,aa);
liczba_cech_swfit = [liczba_cech_swfit size(xucz,2)]

% save tools_svm xuczr ducz xtestr dtest
%%%SVM
% % save tools_svm xuczr ducz xtestr dtest
% piaty parametr true/false - czy badac, testowac rozne wartosci gamma
[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, false, 0.007) 
%[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, true) 
ySVM1 = [ySVM1 ySVM];
accSVM1 = [accSVM1 accSVM];
errorSVM1 = [errorSVM1 errorSVM];
svmModel1{end+1} = svmModel;
theGammma1 = [theGammma1 theGammma];
Pred_y=[Pred_y  ySVM];
wyn=[wyn  accSVM];


%NCA
nca=fscnca(xu, ducz)
%q2=find(nca.FeatureWeights>0.00000000001);%ISIC
q2=find(nca.FeatureWeights>0.000001);%ISIC
%%%SVM%%%SVM
xucz=xu(:,q2); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q2);
liczba_cech_nca = [liczba_cech_nca size(xucz,2)]

% % save tools_svm xuczr ducz xtestr dtest
% piaty parametr true/false - czy badac, testowac rozne wartosci gamma
[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, false, 0.007) 
%[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, true) 
ySVM2 = [ySVM2 ySVM];
accSVM2 = [accSVM2 accSVM];
errorSVM2 = [errorSVM2 errorSVM];
svmModel2{end+1} = svmModel;
theGammma2 = [theGammma2 theGammma];
Pred_y=[Pred_y  ySVM];
wyn=[wyn  accSVM];


%%
%RELIEFF
[RANKED,WEIGHT] = relieff(xu,ducz,50);
q3=RANKED(1:1000);
%%%SVM
xucz=xu(:,q3); 
xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q3);
liczba_cech_relieff = [liczba_cech_relieff size(xucz,2)]

% % save tools_svm xuczr ducz xtestr dtest
% piaty parametr true/false - czy badac, testowac rozne wartosci gamma
[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, false, 0.007) 
%[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, true) 
ySVM3 = [ySVM3 ySVM];
accSVM3 = [accSVM3 accSVM];
errorSVM3 = [errorSVM3 errorSVM];
svmModel3{end+1} = svmModel;
theGammma3 = [theGammma3 theGammma];
Pred_y=[Pred_y  ySVM];
wyn=[wyn  accSVM];


%%
%MRMR
[idx,scores] = fscmrmr(xu,ducz);
%bar(scores(idx))
q4=idx(1:1000);
xucz=xu(:,q4);
xtest=xt(:,q4);
liczba_cech_mrmr = [liczba_cech_mrmr size(xucz,2)]


% % save tools_svm xuczr ducz xtestr dtest
% piaty parametr true/false - czy badac, testowac rozne wartosci gamma
[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, false, 0.007) 
%[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, true) 
ySVM4 = [ySVM4 ySVM];
accSVM4 = [accSVM4 accSVM];
errorSVM4 = [errorSVM4 errorSVM];
svmModel4{end+1} = svmModel;
theGammma4 = [theGammma4 theGammma];
 % [C,order] = confusionmat(double(dtest),double(ySVM));
Pred_y=[Pred_y  ySVM];
wyn=[wyn  accSVM];


%%
%CHi2
[idx,scores] = fscchi2(xu,ducz);
 % bar(scores(idx))
q5=idx(1:1000);
xucz=xu(:,q5);
%xucz=xucz+0.031*randn(size(xucz)).*xucz;
xtest=xt(:,q5);
liczba_cech_chi2 = [liczba_cech_chi2 size(xucz,2)]


% % save tools_svm xuczr ducz xtestr dtest

% piaty parametr true/false - czy badac, testowac rozne wartosci gamma
[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, false, 0.007) 
%[ySVM, accSVM, errorSVM, svmModel, theGammma] = do_svm(xucz, ducz, xtest, dtest, true) 
ySVM5 = [ySVM5 ySVM];
accSVM5 = [accSVM5 accSVM];
errorSVM5 = [errorSVM5 errorSVM];
svmModel5{end+1} = svmModel;
theGammma5 = [theGammma5 theGammma];
Pred_y=[Pred_y  ySVM];
wyn=[wyn  accSVM];


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

     
table(wyn')

metrics_names = {'nclass','dokl_zesp','median_zesp','max_zesp','mean_zesp','odch_zesp'}';
metrics_names = string(metrics_names)
metrics_values = [nclass,dokl_zesp,median_zesp,max_zesp,mean_zesp,odch_zesp]';
table_metrics = table(metrics_names, metrics_values)


table_metrics = table(nclass,dokl_zesp,median_zesp,max_zesp,mean_zesp,odch_zesp)
