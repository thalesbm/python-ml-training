import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset.training_dataset import create_training_dataset

class LogisticRegressionAlgorithm:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def training_model(self):
        print("training_model()")

        X_treino, X_teste, y_treino, y_teste, X, y = create_training_dataset(df=self.dataset)

        cat_cols = X.columns.tolist()

        preproc = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        ohe.fit(X_treino[cat_cols])

        pipe = Pipeline([
            ("prep", preproc),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
        ])

        pipe.fit(X_treino, y_treino)
        y_pred = pipe.predict(X_teste)

        print("=== Logistic Regression (com OHE) ===")
        print("Accuracy:", accuracy_score(y_teste, y_pred))
        print("F1-macro:", f1_score(y_teste, y_pred, average="macro"))
        print(confusion_matrix(y_teste, y_pred))
        print(classification_report(y_teste, y_pred, digits=3))
