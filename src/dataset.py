import pandas as pd

def load() -> pd.DataFrame:
    print("load()")

    path = "files/world_fifa_worldcup_matches.csv"
    dataset: pd.DataFrame = pd.read_csv(path)

    dataset = clear_dataset(dataset)

    print_dataset_info(dataset)

    return dataset

def clear_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    print("clear_dataset()")

    dataset = dataset[[
        "year", 
        "stage_name", 
        "group_name", 
        "country_name",
        "team_a_name", 
        "team_b_name",
        "team_a_code",
        "team_b_id_code",
        "team_a_score", 
        "team_b_score",
        "team_a_win", 
        "team_b_win",
        "draw",
    ]]

    dataset = dataset.rename({"team_b_id_code": "team_b_code"}, axis = 1)
    
    dataset.dropna()

    for nome_col in dataset.columns:
        dataset = dataset[dataset[nome_col].notnull()]

    return dataset

def print_dataset_info(dataset: pd.DataFrame) -> pd.DataFrame:
    print("print_dataset_info()")

    for nome_col in dataset.columns:
        print("--------------------------------------------------")
        print(dataset[nome_col].unique())
        print(dataset[nome_col].value_counts())

    print("--------------------------------------------------")
    print(dataset.head(2))

    print("--------------------------------------------------")
    print(dataset.info())

    print("--------------------------------------------------")
    print(dataset.columns)