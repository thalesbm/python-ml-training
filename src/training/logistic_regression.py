import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset.training_dataset import create_training_dataset

class LogisticRegressionAlgorithm:
    
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def training_model(self):
        print("training_model()")

        X_treino, X_teste, y_treino, y_teste = create_training_dataset(dataset=self.dataset)
        
        pipe = self.build_classifier_pipeline()
        pipe.fit(X_treino, y_treino)

        y_hat = pipe.predict(X_teste)
        y_proba = pipe.predict_proba(X_teste)

        print("--------------------------------")
        print("Metricas Logistic Regression:")
        print("Parametros: ", pipe.get_params())
        print("ACC:", round(accuracy_score(y_teste, y_hat), 3))
        print("| F1-macro:", round(f1_score(y_teste, y_hat, average="macro"), 3))
        print(classification_report(y_teste, y_hat, digits=3))
        print("Matriz de confusão (linhas=verdade, colunas=pred):", confusion_matrix(y_teste, y_hat, labels=["1","2","3"]))
        print("--------------------------------")

        joblib.dump(pipe, "model/logistic_regression.pkl")

    def build_classifier_pipeline(self):
        cat_cols = ["stage_name", "team_a_code", "team_b_code"]
        
        preproc = ColumnTransformer([
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ])
        
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight="balanced")
        pipe = Pipeline([("prep", preproc), ("clf", clf)])
        return pipe