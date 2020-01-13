import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes, svm, linear_model
from sklearn import neighbors
from sklearn.metrics import accuracy_score


# Use five basic machine learning prediction models to predict:
def train_models(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators = 200, random_state=0)
    elif model_name == "Naive Bayes":
        model = naive_bayes.MultinomialNB()
    elif model_name == "Logistic Regression":
        model = linear_model.LogisticRegression(penalty = 'l2')
    elif model_name == "SVM":
        model = svm.SVC(kernel = 'linear')
    elif model_name == "KNN":
        model = neighbors.KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)
    return model


# Use the model to predict and gives accuracy:s
def make_prediction(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    return predictions, accuracy
