import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset.training_dataset import create_training_dataset
from evals.evaluate import evaluate

class RandomForestAlgorithm:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def training_model(self):
        print("training_model()")

        X_treino, X_teste, y_treino, y_teste, X, y = create_training_dataset(df=self.dataset)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample"
        )

        rf.fit(X_treino, y_treino)

        y_pred = rf.predict(X_teste)
        acc = accuracy_score(y_teste, y_pred)
        f1m = f1_score(y_teste, y_pred, average="macro")

        print("\n=== Random Forest (sem OHE) ===")
        print("Accuracy:", acc)
        print("F1-macro:", f1m)
        print(confusion_matrix(y_teste, y_pred))
        print(classification_report(y_teste, y_pred, digits=3))

        self._save_model(list(X_treino.columns), rf, acc, f1m)

    def _save_model(self, columns, rf, acc, f1m):
        print("_save_model()")

        artifact = {
            "model": rf,
            "feature_names": columns,
            "target_name": "salary",
            "metrics": {"accuracy": float(acc), "f1_macro": float(f1m)},
        }

        path = "model/salary/rf_model.joblib"
        joblib.dump(artifact, path)

        art = joblib.load(path)
        artifactory = art["model"]
        feature_names = art["feature_names"]

        evaluate(artifactory, feature_names)
