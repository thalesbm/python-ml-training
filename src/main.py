import pandas as pd
from dataset import load
from graphs import display_graph
from training import make_supervised

def main():
    print("Iniciando aplicação")
    
    dataset = load()

    # display_graph(dataset=dataset)

    make_supervised(dataset=dataset)

if __name__ == "__main__":
    main()
