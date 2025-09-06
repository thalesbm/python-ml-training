import numpy as np
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

def training(dataset: pd.DataFrame):
    print("training()")

    X_treino, X_teste, y_treino, y_teste = _create_training_dataset(dataset=dataset)

    X_treino_scaled, X_teste_scaled = _padronize_dataset(X_treino=X_treino, X_teste=X_teste)

    _compare_graphs(X_treino=X_treino, X_treino_scaled=X_treino_scaled)

    print(X_treino_scaled)


def _create_training_dataset(dataset: pd.DataFrame):
    print("_create_training_dataset()")

    FEATURES = ["stage_name", "team_a_code", "team_b_code"]
    TARGET = "outcome"

    X: pd.DataFrame = dataset[FEATURES].copy()
    y: pd.DataFrame = dataset[TARGET].copy()

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

    print(f"Shape de X_treino :{X_treino.shape}")
    print(f"Shape de X_teste: {X_teste.shape}")
    print(f"Shape de y_treino: {y_treino.shape}")
    print(f"Shape de y_teste: {y_teste.shape}")

    return X_treino, X_teste, y_treino, y_teste

def _padronize_dataset(X_treino: pd.DataFrame, X_teste: pd.DataFrame):
    print("_padronize_dataset()")

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)
    pickle.dump(scaler, open('model/dsa_scaler.pkl','wb'))

    return X_treino_scaled, X_teste_scaled

def _compare_graphs(X_treino: pd.DataFrame, X_treino_scaled: pd.DataFrame):
    print("_compare_graphs()")

    fig, ax = plt.subplots(1, 2, figsize = (15, 5))

    sns.boxplot(data = X_treino, ax = ax[0])
    ax[0].set_title('X_treino Antes da Padronização')

    sns.boxplot(data = X_treino_scaled, ax = ax[1])
    ax[1].set_title('X_treino Depois da Padronização')

    plt.show()
