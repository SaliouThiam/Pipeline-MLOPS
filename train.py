import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Chargement des données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Définition du modèle
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

def get_model():
    return clf

def get_test_data():
    return X_test, y_test

def get_accuracy():
    return acc

# Exécution principale
if __name__ == "__main__":
    mlflow.set_experiment("iris_experiment")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("accuracy", acc)
         # 🔁 Sauvegarde locale
        joblib.dump(clf, "model.joblib")

        # 📦 Sauvegarde dans MLflow UI (comme artefact)
        mlflow.log_artifact("model.joblib")
        
        mlflow.sklearn.log_model(clf, "iris_model")
        

        
    print(f"Accuracy: {acc:.3f}")
