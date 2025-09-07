import numpy as np
import pandas as pd

def load() -> pd.DataFrame:
    print("load()")

    dataset = _load_dataset()
    dataset = _clear_dataset(dataset=dataset)
    dataset = _finding_winner(dataset=dataset)
    dataset = _convert_dataset(dataset=dataset)

    print(dataset.head(10))

    return dataset

def _load_dataset() -> pd.DataFrame:
    print("_load_dataset()")

    path = "files/world_fifa_worldcup_matches.csv"
    dataset: pd.DataFrame = pd.read_csv(path)
    return dataset

def _clear_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    print("_clear_dataset()")

    dataset = dataset[[
        "stage_name",
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
        ["1", "2", "3"],
        default="NA" 
    )

    return dataset

def _convert_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    print("_convert_dataset()")

    stage_name_columns = {
        'group stage': 0, 'second group stage': 1, 'round of 16': 2, 'quarter-finals': 3, 'semi-finals': 4, 'final': 5, 
        'third-place match': 6, 'final round': 7,
    }
    dataset['stage_name'] = dataset['stage_name'].map(stage_name_columns)
    
    team_name_columns = {
        'FRA': 1, 'USA': 2, 'YUG': 3, 'ROU': 4, 'ARG': 5, 'CHL': 6, 'URY': 7, 'BRA': 8, 'PRY': 9, 'AUT': 10,
        'CSK': 11, 'DEU': 12, 'HUN': 13, 'ITA': 14, 'ESP': 15, 'SWE': 16, 'CHE': 17, 'CUB': 18, 'ENG': 19, 'TUR': 20,
        'NIR': 21, 'SUN': 22, 'MEX': 23, 'WAL': 24, 'PRT': 25, 'PRK': 26, 'PER': 27, 'BEL': 28, 'BGR': 29, 'DDR': 30,
        'COD': 31, 'POL': 32, 'AUS': 33, 'SCO': 34, 'NLD': 35, 'HTI': 36, 'TUN': 37, 'DZA': 38, 'HND': 39, 'CAN': 40,
        'MAR': 41, 'KOR': 42, 'IRQ': 43, 'DNK': 44, 'ARE': 45, 'CRI': 46, 'CMR': 47, 'IRL': 48, 'COL': 49, 'NOR': 50,
        'NGA': 51, 'SAU': 52, 'BOL': 53, 'RUS': 54, 'GRC': 55, 'JAM': 56, 'ZAF': 57, 'JPN': 58, 'HRV': 59, 'CHN': 60,
        'SEN': 61, 'SVN': 62, 'ECU': 63, 'TTO': 64, 'SCG': 65, 'AGO': 66, 'CZE': 67, 'TGO': 68, 'IRN': 69, 'CIV': 70,
        'UKR': 71, 'SRB': 72, 'NZL': 73, 'SVK': 74, 'BIH': 75, 'EGY': 76, 'ISL': 77, 'PAN': 78, 'IDN': 79, 'GHA': 80, 
        'SLV': 81, 'ISR': 82, 'KWT': 83
    }
    dataset['team_a_code'] = dataset['team_a_code'].map(team_name_columns)
    dataset['team_b_code'] = dataset['team_b_code'].map(team_name_columns)

    return dataset
