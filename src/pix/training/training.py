from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import joblib

def training_model(X_train, y_train):
    print("training_model()")

    pipeline = Pipeline([
        ("feats", FeatureUnion([
            # ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", 
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                C=1.0
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

def evals_model(pipeline, X_test, y_test):
    print("evals()")

    y_pred = pipeline.predict(X_test)
    
    print("########################################")
    print("Dados do Teste do Treinamento")
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatorio de Classificação:", classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("########################################")

def save(pipeline):
    print("save()")

    path = "model/pix/pix_saldo1.joblib"
    joblib.dump(pipeline, path)

    return joblib.load(path)