from collections import Counter
import json
import re, unicodedata

def test(model):
    print("test()")

    data = _load_file()

    samples = _load_phrases(data)

    _count_intentions(data, samples)

    _validate(model, samples)

def _load_file():
    print("_load_file()")
    
    with open("files/pix/dataset-calcada.json", "r", encoding="utf-8") as f:
        data = json.load(f) 
    return data

def _load_phrases(data):
    print("_load_phrases()")

    samples = [
        item["text"]
        for item in data
        if isinstance(item, dict)
        and "text" in item
    ]
    samples = list(dict.fromkeys(samples))

    return samples

def _count_intentions(data, samples):
    print("_count_intentions()")

    text_to_intent = {}
    for item in data:
        if isinstance(item, dict) and "text" in item and "intent" in item:
            k = item["text"]
            if k not in text_to_intent:
                text_to_intent[k] = str(item["intent"]).strip().lower()

    counts = Counter(text_to_intent.get(t, None) for t in samples)
    counts.pop(None, None)

    qtd_pix = counts.get("pix", 0)
    qtd_saldo = counts.get("saldo", 0)
    qtd_limite = counts.get("limite", 0)

    print("frases de pix:", qtd_pix)
    print("frases de saldo:", qtd_saldo)
    print("frases de limite:", qtd_limite)
    print("total de frases:", len(samples))

def _validate(model, samples):
    print("_validate()")

    preds = model.predict(samples)
    proba = model.predict_proba(samples)
    categories = list(model.classes_)

    saldo = 0
    pix = 0
    limite = 0

    for i, y in enumerate(preds):
        j = categories.index(y)
        prob = float(proba[i, j])

        print(f"{prob:.4f} | {y} | {samples[i]}")

        if y == "saldo":
            saldo = saldo + 1
        elif y == "pix":
            pix = pix + 1
        elif y == "limite":
            limite = limite + 1

    print("Resultado Final:", pix)
    print("pix:", pix)
    print("saldo:", saldo)
    print("limite:", limite)
