%% Load Data
load('C:\Users\namneet\Documents\kit\train\images_train.txt');
images=csvread('C:\Users\namneet\Documents\kit\train\images_train.txt');
load('C:\Users\namneet\Documents\kit\train\genders_train.txt')
y=csvread('C:\Users\namneet\Documents\kit\train\genders_train.txt');
load('C:\Users\namneet\Documents\kit\test\images_test.txt');
imagesTest=csvread('C:\Users\namneet\Documents\kit\test\images_test.txt');
load('C:\Users\namneet\Documents\kit\train\words_train.txt')
x=csvread('C:\Users\namneet\Documents\kit\train\words_train.txt');
load('C:\Users\namneet\Documents\kit\train\image_features_train.txt');
image_features_train=csvread('C:\Users\namneet\Documents\kit\train\image_features_train.txt');
load('C:\Users\namneet\Documents\kit\test\words_test.txt')
x_test=csvread('C:\Users\namneet\Documents\kit\test\words_test.txt');
load('C:\Users\namneet\Documents\kit\test\image_features_test.txt');
image_features_test=csvread('C:\Users\namneet\Documents\kit\test\image_features_test.txt');

addpath ./libsvm;
kIntersect = @(x1,x2) kernel_intersection(x1, x2);

%% Grayscale and Pca on images (Train and test)
[NumObs,NumFeatures]=size(images);
grayscale = zeros(size(images));
for i=1:NumObs
    grayscale(i,:)= mat2gray(images(i,:));
end
[~,scaledImages]=pca(grayscale, 'NumComponents', 30);

[NumObsTest,NumFeaturesTest]=size(imagesTest);
grayscaleTest = zeros(size(imagesTest));
for i=1:NumObsTest
    grayscaleTest(i,:)= mat2gray(imagesTest(i,:));
end
[~,scaledImagesTest]=pca(grayscaleTest, 'NumComponents', 30);

%% Most Frequent words for each gender
females=x((find(y==1)),:);
f1=sum(females);
u=unique(f1);
%features=sort(u, 'descend');
femalewords=find(f1>=3000);

males=x((find(y==0)),:);
m1=sum(males);
u_males=unique(m1);
%features_males=sort(u_males, 'descend');
malewords=find(m1>=3000);

%% Kernel Intersection on everything
features=[femalewords malewords];
Train1=[x(:,unique(features)) image_features_train scaledImages];
Test1=[x_test(:,unique(features)) image_features_test scaledImagesTest];
Ytest1 = zeros(size(Test1,1),1);
[predict1, model1] = kernel_libsvm(Train1, y, Test1, Ytest1, kIntersect);

%% Kernel intersection on words
Train2=x(:,unique(features));
Test2=x_test(:,unique(features));
Ytest2 = zeros(size(Test2,1),1);
[predict2, model2] = kernel_libsvm(Train2, y, Test2, Ytest2, kIntersect);

%% Kernel intersection on words and two image features
Train3=[x(:,unique(features)) image_features_train(:,1:2)];
Test3=[x_test(:,unique(features)) image_features_test(:,1:2)];
Ytest3 = zeros(size(Test2,1),1);
[predict3, model3] = kernel_libsvm(Train3, y, Test3, Ytest3, kIntersect);

%% Ensemble: adaboostm1
model4=fitensemble(Train1,y,'AdaBoostM1',1000, 'tree');
predict4=predict(model4,testdata); 
