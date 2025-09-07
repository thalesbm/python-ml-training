import pandas as pd

def evaluate(model, feature_names):
    print("evaluate()")

    test = {
        "age": 3,                # 31-40 anos → mapeado como 2
        "workclass": 1,          # private
        "education": 2,          # bachelors
        "race": 0,               # white
        "sex": 0,                # male
        "native-country": 0,     # united-states
    }

    X_new = pd.DataFrame([test], columns=feature_names)

    X_new = pd.DataFrame([test])

    # Predição
    # 0 = <=50K, 1 = >50K
    classe = model.predict(X_new)[0]  

    # probabilidade de >50K
    proba = model.predict_proba(X_new)[0,1]

    faixa = "<=50K" if classe == 0 else ">50K"
    print(f"Faixa salarial prevista: {faixa} (prob. {proba:.2f})")
