part = make_partition_CV(size(words_train,1), 6);

w=words_train;
x=features_train;
y=genders_train;

n=max(part);
for total=1:n
a=1;
b=1;
words_test_final=[];
features_test_final=[];
ytest_final=[];
words_train_final=[];
ytrain_final=[];
features_train_final=[];

    for g=1:4998
     if part(g)==total
            words_test_final(a,:)=w(g,:);
            features_test_final(a,:)=x(g,:);
            ytest_final(a,:)=y(g,:);
            a=a+1;
        else
            words_train_final(b,:)=w(g,:);
            features_train_final(b,:)=x(g,:);
            ytrain_final(b,:)=y(g,:);
            b=b+1;
     end
    end    
       
    SVMModel = fitcsvm(words_train_final, ytrain_final);
    [testLabels_train_words,score] = predict(SVMModel, words_test_final);
    
    SVMModel1 = fitcsvm(features_train_final, ytrain_final);
    [testLabels_train_features,score] = predict(SVMModel1, features_test_final);
    
    combo = [testLabels_train_words, testLabels_train_features];
    
    ctree = fitctree(combo, ytrain_final);
    testLabels_train = predict(ctree, test_final);    
    
error1= mean(sign(testLabels_train)~=sign(ytest_final));
errorinit(total,1)=error1;
end

error=mean(errorinit);