%% Load Data
load('/Users/namneet/Desktop/CIS520/kit/train/images_train.txt');
images=csvread('/Users/namneet/Desktop/CIS520/kit/train/images_train.txt');
load('/Users/namneet/Desktop/CIS520/kit/train/genders_train.mat')
y=genders_train;
load('/Users/namneet/Desktop/CIS520/kit/test/images_test.txt');
imagesTest=csvread('/Users/namneet/Desktop/CIS520/kit/test/images_test.txt');
load('/Users/namneet/Desktop/CIS520/kit/train/words_train.mat')
x=words_train;
load('/Users/namneet/Desktop/CIS520/kit/train/image_features_train.mat');
load('/Users/namneet/Desktop/CIS520/kit/test/words_test.mat')
x_test=words_test;
load('/Users/namneet/Desktop/CIS520/kit/test/image_features_test.mat');

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
[~,scaledImagesTest]=pca(grayscaleTest, 'NumCompoents', 30);

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

%% Feature matrix
features=[femalewords malewords];
xhat=[x(:,unique(features)) image_features_train scaledImages];
xhat_test=[x_test(:,unique(features)) image_features_test scaledImagesTest];

%% Kernel Intersection
kIntersect = @(x1,x2) kernel_intersection(x1, x2);

K = kIntersect(xhat, xhat);
Ktest = kIntersect(xhat, xhat_test);
Ytest = zeros(size(x_test,1),1);

%% Use built-in libsvm cross validation to choose the C regularization parameter.
crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

%% Train and evaluate SVM classifier using libsvm
model = svmtrain(y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);