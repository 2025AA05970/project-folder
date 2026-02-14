from sklearn.linear_model import LogisticRegression

def train_predict(X_train, X_test, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return y_pred, y_prob
