from dataset.dataset import load
from training.decision_tree import DecisionTreeAlgorithm
from training.linear_regression import LinearRegressionAlgorithm 
from training.random_forest import RandomForestAlgorithm
from training.logistic_regression import LogisticRegressionAlgorithm

import pandas as pd

def init():
    print("Iniciando aplicação")
    
    dataset = load()
    DecisionTreeAlgorithm(dataset=dataset).training_model()
    LinearRegressionAlgorithm(dataset=dataset).training_model()
    RandomForestAlgorithm(dataset=dataset).training_model()
    LogisticRegressionAlgorithm(dataset=dataset).training_model()

if __name__ == "__main__":
    init()
