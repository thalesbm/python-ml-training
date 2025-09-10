def test(model, samples):
    preds = model.predict(samples)
    proba = model.predict_proba(samples)
    categories = list(model.classes_)

    for i, y in enumerate(preds):
        j = categories.index(y)
        prob = float(proba[i, j])

        print(f"{prob:.4f} | {y} | {samples[i]}")
