import pandas as pd
from dataset.dataset import load
from graphs import display_graph
from training import training

def main():
    print("Iniciando aplicação")
    
    dataset = load()

    # display_graph(dataset=dataset)

    training(dataset=dataset)

if __name__ == "__main__":
    main()
