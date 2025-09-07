from dataset.dataset import load

from training.logistic_regression import LogisticRegressionAlgorithm
from training.random_forest import RandomForestAlgorithm

def init():
    print("Iniciando aplicação")

    dataset = load()

    # LogisticRegressionAlgorithm(dataset=dataset).training_model()
    RandomForestAlgorithm(dataset=dataset).training_model()

if __name__ == "__main__":
    init()
