data_pooled = images_train;
labels_pooled = genders_train;

dim= 20:25;

[coeff,score,latent]=pca(data_pooled);

%extracting train and test scores from the score matrix obtained above
train_score = score(1:3500,:);
teststartindx=3500;
test_score = score(teststartindx+1:end,:);

for i = 1:length(dim)
    
    %Number of top dimensions considered in the particular iteration i
    numdim = dim(i);
    
    %Extracting the principle scores from the scores matrix based on the
    %number of top dimensions considered
    principle_score = train_score(:,1:numdim);
    
    %Fitting a Naive Bayes model to the training data represented by
    %principle_score and classes represented by trainset.letter vector
    model = NaiveBayes.fit(principle_score, labels_pooled(1:3500));
    
    %Making predictions based on the model
    predicted = model.predict(test_score(:,1:numdim));
    
     %Computing the error
    total_err=sum(predicted ~= labels_pooled(3501:5000));
    total_letters_len=length(labels_pooled(3501:5000));
    
    %Computing accuracy from the error
    acc(i)=1-total_err/total_letters_len;
end

%displaying accuracy corresponding to each of the three dimensions
acc