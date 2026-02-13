from sklearn.naive_bayes import GaussianNB

def train_predict(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return y_pred, y_prob
