import sys
import os

# Ajouter le dossier parent au path pour accéder à train.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import get_model, get_test_data, get_accuracy
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    clf = get_model()
    X_test, y_test = get_test_data()
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc > 0.7
