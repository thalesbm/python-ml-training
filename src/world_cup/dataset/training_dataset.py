import pandas as pd

from sklearn.model_selection import train_test_split

def create_training_dataset(dataset: pd.DataFrame):
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
