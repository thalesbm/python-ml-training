import pandas as pd

from dataset.format import normalize_text
from sklearn.metrics import classification_report

def test(model):
    print("test()")

    data = _load_file()

    _validate(model, data)

def _load_file():
    print("_load_file()")
    
    data: pd.DataFrame = pd.read_json("files/pix/teste/dataset_calcada.json")
    data.loc[data["intent"] == "saldo", "intent"] = "outro"
    data.loc[data["intent"] == "limite", "intent"] = "outro"

    data["text"] = data["text"].str.lower()

    # com e sem isso deu o mesmo resultado
    data["text"] = data["text"].astype(str).apply(normalize_text)

    data = data.drop_duplicates(subset=["text"])

    print("########################################")
    print("Total de Dados para Teste")
    print(data["intent"].value_counts())
    print("########################################")

    return data

def _validate(model, data):
    print("_validate()")

    X = data["text"]
    y_true = data["intent"]

    preds = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    erros = 0
    pix_pred = 0
    nao_pix_pred = 0

    for i in range(len(X)):
        y_hat = preds[i]
        j = classes.index(y_hat)
        p = float(proba[i, j])

        if y_hat != y_true.iloc[i]:
            print(f"Prob: {p:.4f} | Pred: {y_hat} | [ Msg: {X.iloc[i]} | True: {y_true.iloc[i]} ]")
            erros += 1

        if y_hat == "pix":
            pix_pred += 1
        else:
            nao_pix_pred += 1

    print("########################################")
    print("Resultado dos Testes")
    print("pix:", pix_pred)
    print("nao-pix:", nao_pix_pred)
    print("erros:", erros)
    print(f"taxa de erro (erros/N): {erros * 100 / len(X):.2f} %")
    print(classification_report(y_true, preds, digits=3))
    print("########################################")
