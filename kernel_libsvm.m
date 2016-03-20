[coeff1, score1, latent1] = pca(words_train, 'NumComponents',100);
X_hat_train = (coeff1 * score1')';

[coeff, score, latent] = pca(words_test, 'NumComponents',100);
X_hat_test = (coeff * score')';

kIntersect = @(x1,x2) kernel_intersection(x1, x2);
Y=genders;

K = kIntersect(X_hat_train, X_hat_train);
Ktest = kIntersect(X_hat_train, X_hat_test);
Ytest = zeros(size(X_hat_test,1),1);

% Use built-in libsvm cross validation to choose the C regularization
% parameter.
crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);

