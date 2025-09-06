import numpy as np
import pandas as pd

def load() -> pd.DataFrame:
    print("load()")

    dataset = _load_dataset()
    dataset = _clear_dataset(dataset=dataset)
    dataset = _finding_winner(dataset=dataset)

    _print_dataset_info(dataset)

    return dataset

def _load_dataset() -> pd.DataFrame:
    print("_load_dataset()")

    path = "files/world_fifa_worldcup_matches.csv"
    dataset: pd.DataFrame = pd.read_csv(path)
    return dataset

def _clear_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    print("_clear_dataset()")

    dataset = dataset[[
        "year",
        "stage_name",
        "group_name",
        "team_a_code",
        "team_b_id_code",
        "team_a_score", 
        "team_b_score",
        "team_a_win", 
        "team_b_win",
        "draw",
    ]]

    dataset = dataset.rename({"team_b_id_code": "team_b_code"}, axis = 1)
    
    dataset = dataset.dropna()

    for nome_col in dataset.columns:
        dataset = dataset[dataset[nome_col].notnull()]

    return dataset

def _finding_winner(dataset: pd.DataFrame) -> pd.DataFrame:
    print("_finding_winner()")

    cols = ["team_a_win", "draw", "team_b_win"]
    dataset[cols] = dataset[cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    dataset["outcome"] = np.select(
        [dataset["team_a_win"].eq(1), dataset["draw"].eq(1), dataset["team_b_win"].eq(1)],
        ["A", "D", "B"],
        default="NA" 
    )

    return dataset

def _print_dataset_info(dataset: pd.DataFrame) -> pd.DataFrame:
    print("_print_dataset_info()")

    for nome_col in dataset.columns:
        print("--------------------------------------------------")
        print(dataset[nome_col].unique())
        print(dataset[nome_col].value_counts())

    print("--------------------------------------------------")
    print(dataset.info())

    print("--------------------------------------------------")
    print(dataset.columns)

    print("--------------------------------------------------")
    print(dataset.head(10))