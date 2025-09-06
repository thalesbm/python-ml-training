import pandas as pd
from dataset import load
from graphs import display_graph

def main():
    print("Iniciando aplicação")
    
    dataset = load()

    display_graph(dataset)

if __name__ == "__main__":
    main()
