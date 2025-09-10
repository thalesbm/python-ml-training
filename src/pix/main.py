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

    samples = [
        "quero ver meu saldo agora",
        "faz um pix de 50 pro joão",
        "quanto eu tenho de grana sobrando?",
        "manda 200 conto pra maria rapidin",
        "verificar saldo disponível",
        "passa 30 pila pro 11987654321"
    ]

    test(model, samples)

if __name__ == "__main__":
    init()
