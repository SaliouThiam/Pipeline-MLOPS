from train import get_model, get_test_data, get_accuracy
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    clf = get_model()
    X_test, y_test = get_test_data()
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    # Ou simplement : acc = get_accuracy()
    assert acc > 0.7
