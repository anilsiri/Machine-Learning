[coeff, score, latent] = pca(images_train, 'NumComponents',25);
X_hat_train = (coeff * score')';

[coeff1, score1, latent1] = pca(image_test, 'NumComponents', 25);
X_hat1_test = (coeff1 * score1')';

nb5 = fitncb(X_hat_train, genders_train);
image_label = predict(nb5, X_hat1_test);