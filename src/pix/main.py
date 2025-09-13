from dataset.dataset import load_dataset, create_training_dataset
from training.training import training_model, evals_model, save
from evals.evals import test

def init():
    print("Iniciando aplicação")

    # load dataset
    dataset = load_dataset()

    # split training / test dataset
    X_train, X_test, y_train, y_test = create_training_dataset(dataset=dataset)
    
    # training
    pipeline = training_model(X_train, y_train)

    # evals model with test
    evals_model(pipeline, X_test, y_test)

    # save
    model = save(pipeline)

    # teste
    test(model)

    teste = [
        "quero pagar 320 para maria.silva+pagamento@example.com",
        "envia 750 reais para joao.almeida@example.com",
        "deposita 450 reais para 218.889.164-25",
        "efetue 600 para +55 11 99876-5432",
        "me manda 980 para 7cd50935-e87b-4230-a2f9-0884ad00cee1",
        "passa 250 para 585.213.691-68",
        "realizar pagamento de 870 para 7006d095-aab9-4493-bee7-e8d161f83c5b",
        "efetue 410 para +55 21 99711-2233",
        "coloca 520 reais para ana.pereira@dominio.com",
        "preciso enviar 300 para 123.456.789-00",
        "envia 160 reais para +55 31 99123-4567",
        "manda 560 para paulo.souza@example.com",
        "quero pagar 230 para +55 94 94351-6101",
        "deposita 640 reais para 0a6ecc0e-0fff-498b-8a55-6edd4a4bd363",
        "coloca 700 reais para fernanda.rocha@example.com",
        "me manda 420 para 3f2a1b44-9c8d-4c2f-8f31-8c4d2b2a7f10",
        "realizar pagamento de 915 para +55 71 99222-3344",
        "manda 510 para 218.889.164-25",
        "efetue 840 para pedro99@outlook.com",
        "passa 690 para +55 41 99555-6677",
        "envia 375 reais para carla.mendes@example.com",
        "deposita 810 reais para 9f6b0a1c-2d34-4e1b-9a7c-12ab34cd56ef",
        "quero pagar 455 para juliana.santos@example.com",
        "efetue 505 para +55 85 99666-7788",
        "coloca 285 reais para 585.213.691-68"
        "Me transfira 800 via pix para Beatriz",
        "Preciso enviar 750 para João"
    ]
    
    predict_list(model, teste)

def predict_list(model, text):
    print(model.predict(text))

if __name__ == "__main__":
    init()
