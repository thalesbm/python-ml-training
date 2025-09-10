from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import joblib

def training_model(X_train, y_train):
    print("training_model()")

    pipeline = Pipeline([
        ("tfidf", 
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3,5),
                min_df=2,
                sublinear_tf=True
            )
        ),
        ("clf", 
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced"
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

def evals_model(pipeline, X_test, y_test):
    print("evals()")

    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

def save(pipeline):
    print("save()")

    path = "model/pix/pix_saldo.joblib"
    joblib.dump(pipeline, path)

    return joblib.load(path)  