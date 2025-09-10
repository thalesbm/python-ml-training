import numpy as np
import pandas as pd

from dataset.dataset import load_dataset, create_training_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def training_model(X_train, y_train):
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

def evals(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))    

def init():
    print("Iniciando aplicação")

    dataset = load_dataset()
    X_train, X_test, y_train, y_test = create_training_dataset(dataset=dataset)
    
    pipeline = training_model(X_train, y_train)

    evals(pipeline, X_test, y_test)

if __name__ == "__main__":
    init()
