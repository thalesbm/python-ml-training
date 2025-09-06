import pandas as pd

def main():
    print("Iniciando aplicação")

    dataset = pd.read_csv("files/mundo_transfermarkt_competicoes_brasileirao_serie_a.csv")

    print(dataset.columns) 
    print(dataset.head())

if __name__ == "__main__":
    main()

