from sklearn.neighbors import KNeighborsClassifier

def train_predict(X_train, X_test, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return y_pred, y_prob
