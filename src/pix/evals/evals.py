import pandas as pd

from dataset.format import normalize_text

def test(model):
    print("test()")

    data = _load_file()

    _validate(model, data)

def _load_file():
    print("_load_file()")
    
    data: pd.DataFrame = pd.read_json("files/pix/teste/dataset_calcada.json")
    data = data.drop_duplicates(subset=["text"])
    data.loc[data["intent"] == "saldo", "intent"] = "outro"
    data.loc[data["intent"] == "saldo", "intent"] = "outro"

    data["text"] = data["text"].str.lower()

    data["text"] = data["text"].astype(str).apply(normalize_text)

    print(data["intent"].value_counts())

    return data

def _validate(model, data):
    print("_validate()")

    preds = model.predict(data["text"])
    proba = model.predict_proba(data["text"])
    categories = list(model.classes_)

    error = 0
    pix = 0
    nao_pix = 0

    for i, y in enumerate(preds):
        j = categories.index(y)
        prob = float(proba[i, j])

        try:
            if y != data["intent"][i]:
                print(f"Prob: {prob:.4f} | Categoria: {y} | [ Mensagem: {data["text"][i]} | Categoria: {data["intent"][i]} ]")

            if y == "pix":
                pix = pix + 1
            else:
                nao_pix = nao_pix + 1
        except Exception:
            error = error + 1

    print("pix:", pix)
    print("nao_pix:", nao_pix)
    print("error:", error)
    print(f"taxa de erro: {(nao_pix * 100 / pix):.2f} %")
