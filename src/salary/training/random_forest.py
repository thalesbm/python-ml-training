import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset.training_dataset import create_training_dataset

class RandomForestAlgorithm:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def training_model(self):
        X_treino, X_teste, y_treino, y_teste, X, y = create_training_dataset(df=self.dataset)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample"
        )

        rf.fit(X_treino, y_treino)
        y_pred_rf = rf.predict(X_teste)

        print("\n=== Random Forest (sem OHE) ===")
        print("Accuracy:", accuracy_score(y_teste, y_pred_rf))
        print("F1-macro:", f1_score(y_teste, y_pred_rf, average="macro"))
        print(confusion_matrix(y_teste, y_pred_rf))
        print(classification_report(y_teste, y_pred_rf, digits=3))
