import numpy as np
import pandas as pd

from dataset.format import normalize_text
from sklearn.model_selection import train_test_split

def load_dataset() -> pd.DataFrame:
    print("load_dataset()")

    dfs = []
    for path in [
        "files/pix/dataset_pix.json",
        "files/pix/dataset_saldo.json",
        "files/pix/dataset_pix_saldo.json",
        "files/pix/dataset_pix_saldo_girias.json",
        "files/pix/dataset_limite_girias.json",
        "files/pix/dataset_limite.json",
        "files/pix/dataset_limite_produtos.json",
        "files/pix/dataset_pix_2.json",
        "files/pix/dataset_pix_diferentes_chaves.json",
    ]:
        dfs.append(pd.read_json(path))

    df = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.loc[df["Intenção"].isin(["saldo", "limite"]), "Intenção"] = "outro"

    df["Mensagem"] = df["Mensagem"].astype(str).apply(normalize_text)

    df = df.drop_duplicates(subset=["Mensagem"])

    print(df["Intenção"].value_counts())

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
