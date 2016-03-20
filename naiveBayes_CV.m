part = make_partition_CV(size(features_train,1), 10);
x=words_train;
y=genders;
n=max(part);

for total=1:n
a=1;
b=1;
test=[];
ytest=[];
train=[];
ytrain=[];
    for g=1:4998
     if part(g)==total
            test(a,:)=x(g,:);
            ytest(a,:)=y(g,:);
            a=a+1;
        else
            train(b,:)=x(g,:);
            ytrain(b,:)=y(g,:);
            b=b+1;
     end
    end
    
    [coeff, score, latent] = pca(train, 'NumComponents', 1000);
    X_hat_train = (coeff * score')';

    SVMModel = fitcsvm(X_hat_train, ytrain);
    [testLabels,score] = predict(SVMModel, test);
    
    error1= mean(sign(testLabels)~=sign(ytest));
    errorinit(total,1)=error1;
end
error=mean(errorinit);