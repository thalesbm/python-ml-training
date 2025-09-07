import numpy as np
import pandas as pd

def load() -> pd.DataFrame:
    print("load()")

    df = _load_dataset()
    df = _clear_column_workclass(df=df)
    df = _clear_column_sex(df=df)
    df = _clear_column_age(df=df)
    df = _clear_column_race(df=df)
    df = _clear_column_country(df=df)
    df = _clear_column_education(df=df)
    df = _clear_column_salary(df=df)

    df = df.astype("float64")

    print(df.head(10))

    return df

def _load_dataset() -> pd.DataFrame:
    print("_load_dataset()")

    path = "../files/adult_outcome/salary.csv"
    df: pd.DataFrame = pd.read_csv(path)

    return df

def _clear_column_workclass(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_workclass()")

    freq = df["workclass"].value_counts(normalize=True).sort_values(ascending=False)

    raras = freq[freq < 0.01].index
    df["workclass"] = df["workclass"].where(~df["workclass"].isin(raras), "others")

    df["workclass"] = (df["workclass"]
            .astype(str).str.strip().str.lower()
            .replace({"others": "others", "?": "others"})
            .replace({"local-gov": "gov", "state-gov": "gov", "federal-gov": "gov"})
            .replace({"self-emp-not-inc": "self-employed", "self-emp-inc": "self-employed"}))

    map_workclass = {'gov': 0, 'self-employed': 1, 'private': 2, 'others': 3}
    df['workclass'] = df['workclass'].map(map_workclass)

    return df

def _clear_column_sex(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_sex()")
    
    map_sex = {' Male': 0, ' Female': 1}
    df['sex'] = df['sex'].map(map_sex)

    return df

def _clear_column_age(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_age()")
    
    age_column = df["age"].clip(lower=1, upper=100)

    start = ((age_column - 1) // 10) * 10 + 1
    bins = list(range(1, 101, 10)) + [101] 
    labels = [f"{start}-{start+9}" for start in range(1, 100, 10)]

    df["age"] = pd.cut(age_column, bins=bins, right=False, labels=labels, ordered=True)

    df["age"] = (df["age"]
            .astype(str).str.strip().str.lower()
            .replace({"1-10": "11-20", "91-100": "71-80", "81-90": "71-80", "71-80": "61-70"}))

    map_age = {'11-20': 0, '21-30': 1, '31-40': 2, '41-50': 3, '51-60': 4, '61-70': 5}
    df['age'] = df['age'].map(map_age)

    return df

def _clear_column_race(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_race()")

    df["race"] = (df["race"]
            .astype(str).str.strip().str.lower()
            .replace({"asian-pac-islander": "others", "amer-indian-eskimo": "others", "other": "others"}))

    map_race = {'white': 0, 'black': 1, 'others': 2}
    df['race'] = df['race'].map(map_race)

    return df

def _clear_column_country(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_country()")

    df["native-country"] = (df["native-country"]
            .astype(str).str.strip().str.lower()
            .replace({"mexico": "others", "?": "others", "philippines": "others", "china": "others", "japan": "others", "india": "others", "germany": "others", 
                        "canada": "others", "puerto-rico": "others", "el-salvador": "others", "cuba": "others", "england": "others", "jamaica": "others", "south": "others", 
                        "italy": "others", "dominican-republic": "others", "vietnam": "others", "guatemala": "others", "poland": "others", "columbia": "others", "taiwan": "others", 
                        "haiti": "others", "iran": "others", "holand-netherlands": "others", "scotland": "others", "hungary": "others", "honduras": "others", 
                        "yugoslavia": "others", "laos": "others", "thailand": "others", "cambodia": "others", "trinadad&tobago": "others", "hong": "others", "ireland": "others", "ecuador": "others",
                        "greece": "others", "france": "others", "peru": "others", "nicaragua": "others", "outlying-us(guam-usvi-etc)": "others",}))

    map_native_country = {'united-states': 0, 'others': 1}
    df['native-country'] = df['native-country'].map(map_native_country)

    return df

def _clear_column_education(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_education()")

    df["education"] = (df["education"]
            .astype(str).str.strip().str.lower()
            .replace({"11th": "school", "10th": "school", "7th-8th": "school", "9th": "school", "12th": "school", "1st-4th": "school", "5th-6th": "school"})
            .replace({"assoc-voc": "assoc", "assoc-acdm": "assoc"})
            .replace({"preschool": "school", "prof-school": "school"}))


    map_education = {'hs-grad': 0, 'some-college': 1, 'bachelors': 2, 'school': 3, 'assoc': 4, 'masters': 5, 'doctorate': 6}
    df['education'] = df['education'].map(map_education)

    return df

def _clear_column_salary(df: pd.DataFrame) -> pd.DataFrame:
    print("_clear_column_salary()")

    map_salary = {' <=50K': 0, ' >50K': 1}
    df['salary'] = df['salary'].map(map_salary)

    return df
