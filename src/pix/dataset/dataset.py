import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_dataset() -> pd.DataFrame:
    print("load_dataset()")

    # 7000 frases
    path = "files/pix/pix_intencoes_final.json"
    df_pix: pd.DataFrame = pd.read_json(path)

    # 5000 frases
    path = "files/pix/saldo_intencoes.json"
    df_balance: pd.DataFrame = pd.read_json(path)

    # 6000 frases genericas
    path = "files/pix/nlu_intencoes_naturais.json"
    df_random: pd.DataFrame = pd.read_json(path)

    # 2000 frases com girias
    path = "files/pix/nlu_intencoes_girias.json"
    df_girias: pd.DataFrame = pd.read_json(path)

    df = pd.concat([df_pix, df_balance, df_random, df_girias], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df

def create_training_dataset(dataset: pd.DataFrame):
    print("create_training_dataset()")

    X = dataset["Mensagem"]
    y = dataset["Intenção"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Treino:", len(X_train), "Teste:", len(X_test))
    print("Distribuição treino:", y_train.value_counts(normalize=True))
    print("Distribuição teste:", y_test.value_counts(normalize=True))
    print(f"Shape de X_treino :{X_train.shape}")
    print(f"Shape de X_teste: {X_test.shape}")
    print(f"Shape de y_treino: {y_train.shape}")
    print(f"Shape de y_teste: {y_test.shape}")

    return X_train, X_test, y_train, y_test