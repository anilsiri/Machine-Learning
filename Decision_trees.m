%function [error] = knn_xval_error(K, X, Y, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
% FILL IN YOUR CODE HERE

part = make_partition_CV(size(words_train,1), 2);

n=max(part);
for total=1:n
a=1;
b=1;
test_final=[];
ytest_final=[];
train_final=[];
ytrain_final=[];
    for g=1:4998
     if part(g)==total
            test_final(a,:)=x(g,:);
            ytest_final(a,:)=y(g,:);
            a=a+1;
        else
            train_final(b,:)=x(g,:);
            ytrain_final(b,:)=y(g,:);
            b=b+1;
     end
    end
    
    ctree = fitctree(train_final, ytrain_final);
    newtree = prune(ctree, 'Level', 1);
    testLabels_train = predict(newtree, test_final);    
    error1= mean(sign(testLabels_train)~=sign(ytest_final));
    errorinit(total,1)=error1;
end
error=mean(errorinit);