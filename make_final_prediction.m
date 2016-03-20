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
tp=tic;
add ./libsvm;
words_test=X_test(:,1:5000);
images_test=X_test(:,5001:35000);
image_features_test=X_test(:,35001:35007);

[NumObsTest,NumFeaturesTest]=size(images_test);
grayscaleTest = zeros(size(images_test));
for i=1:NumObsTest
    grayscaleTest(i,:)= mat2gray(images_test(i,:));
end
[~,scaledImagesTest]=pca(grayscaleTest, 'NumComponents', 30);

Test1=[words_test(:,unique(model.features)) image_features_test(:,1:2) scaledImagesTest];
model1=model.model1;
model2=model.mode2;
model4=model.mode4;
predict1=svmpredict(model1,Test1);
predict2=predict(model2,Test1); 
combinedtest=[predict1 predict2];
predictions=predict(model4,combinedtest);
time=toc(tp)
end