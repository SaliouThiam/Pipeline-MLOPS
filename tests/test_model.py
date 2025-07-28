from train import clf, X_test, y_test
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc > 0.7  # le modèle doit atteindre au moins 70%
