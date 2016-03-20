function predictions = make_final_prediction(model,X_train, X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% % Sample model
% predictions = X_test(:,1:5000) * model.w_ridge;
% predictions(predictions > 0.5) = 1;
% predictions(predictions <= 0.5) = 0;

%% load data
addpath ./libsvm;

words_train=X_train(:,1:5000);
images_train=X_train(:,5001:35000);
image_features_train=X_train(:,35001:35007);

words_test=X_test(:,1:5000);
images_test=X_test(:,5001:35000);
image_features_test=X_test(:,35001:35007);
Ytest = zeros(size(X_test,1),1);

kIntersect = @(x1,x2) kernel_intersection(x1, x2);

%% feature selection

[NumObs,NumFeatures]=size(images_train);
grayscale = zeros(size(images_train));
for i=1:NumObs
    grayscale(i,:)= mat2gray(images_train(i,:));
end
[~,scaledImages]=pca(grayscale, 'NumComponents', 30);

[NumObsTest,NumFeaturesTest]=size(images_test);
grayscaleTest = zeros(size(images_test));
for i=1:NumObsTest
    grayscaleTest(i,:)= mat2gray(images_test(i,:));
end
[~,scaledImagesTest]=pca(grayscaleTest, 'NumComponents', 30);


%% prediction
Train1=[words_train(:,unique(model.features)) image_features_train(:,1:2) scaledImages];
Test1=[words_test(:,unique(model.features)) image_features_test(:,1:2) scaledImagesTest];
Ktest = kIntersect(Train1, Test1);
model1=model.model1;
model2=model.mode2;
model4=model.mode4;
[predict1,~,~] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model1);
predict2=predict(model2,Test1); 
combinedtest=[predict1 predict2];
predictions=predict(model4,combinedtest);
end