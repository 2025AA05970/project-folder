from sklearn.ensemble import RandomForestClassifier

def train_predict(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return y_pred, y_prob
