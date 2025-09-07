from world_cup.dataset.dataset import load
from world_cup.training.decision_tree import DecisionTreeAlgorithm
from world_cup.training.linear_regression import LinearRegressionAlgorithm 
from world_cup.training.random_forest import RandomForestAlgorithm
from world_cup.training.logistic_regression import LogisticRegressionAlgorithm

import pandas as pd

def init_world_cup():
    print("Iniciando aplicação")
    
    dataset = load()
    DecisionTreeAlgorithm(dataset=dataset).training_model()
    LinearRegressionAlgorithm(dataset=dataset).training_model()
    RandomForestAlgorithm(dataset=dataset).training_model()
    LogisticRegressionAlgorithm(dataset=dataset).training_model()

def init_covid():
    print("Iniciando aplicação")

    path = "files/adult_outcome/salary.csv"
    dataset: pd.DataFrame = pd.read_csv(path)

    print(dataset.head(10))

    print(dataset.columns)

    print(len(dataset))

if __name__ == "__main__":
    # init_world_cup()
    init_covid()
