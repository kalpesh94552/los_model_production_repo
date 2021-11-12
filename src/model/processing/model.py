from sklearn.naive_bayes import GaussianNB

def naive_bayes_model(X_train, y_train):
    # Naive Bayes
    target = y_train.values
    features = X_train.values
    classifier_nb = GaussianNB()
    model_nb = classifier_nb.fit(features, target)
    
    return model_nb