import numpy as np
import pandas as pd
import pickle
from graphs import compare_graphs

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def training(dataset: pd.DataFrame):
    print("training()")

    X_treino, X_teste, y_treino, y_teste = _create_training_dataset(dataset=dataset)

    X_treino_scaled, X_teste_scaled = _padronize_dataset(X_treino=X_treino, X_teste=X_teste)

    # compare_graphs(X_treino=X_treino, X_treino_scaled=X_treino_scaled)

    _training_model_linear_regression(X_treino_scaled=X_treino_scaled, y_treino=y_treino, X_teste_scaled=X_teste_scaled, y_teste=y_teste)
    
    _training_model_decision_tree(X_treino_scaled=X_treino_scaled, y_treino=y_treino, X_teste_scaled=X_teste_scaled, y_teste=y_teste)

    _training_model_random_forest(X_treino_scaled=X_treino_scaled, y_treino=y_treino, X_teste_scaled=X_teste_scaled, y_teste=y_teste)   

    print("end training()")

def _create_training_dataset(dataset: pd.DataFrame):
    print("_create_training_dataset()")

    FEATURES = ["stage_name", "team_a_code", "team_b_code"]
    TARGET = "outcome"

    X: pd.DataFrame = dataset[FEATURES].copy()
    y: pd.DataFrame = dataset[TARGET].copy()

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

    print(f"Shape de X_treino :{X_treino.shape}")
    print(f"Shape de X_teste: {X_teste.shape}")
    print(f"Shape de y_treino: {y_treino.shape}")
    print(f"Shape de y_teste: {y_teste.shape}")

    return X_treino, X_teste, y_treino, y_teste

def _padronize_dataset(X_treino: pd.DataFrame, X_teste: pd.DataFrame):
    print("_padronize_dataset()")

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)
    pickle.dump(scaler, open('model/dsa_scaler.pkl','wb'))

    return X_treino_scaled, X_teste_scaled

def _training_model_linear_regression(X_treino_scaled: pd.DataFrame, y_treino: pd.DataFrame, X_teste_scaled: pd.DataFrame, y_teste: pd.DataFrame):  
    print("_training_model_linear_regression()")

    model = LinearRegression()
    model.fit(X_treino_scaled, y_treino)

    print(model.get_params())

    y_pred_v1 = model.predict(X_teste_scaled)

    # Métricas
    print('Mean Absolute Error (MAE):', round(mean_absolute_error(y_teste, y_pred_v1),3))  
    print('Root Mean Squared Error (RMSE):', round(np.sqrt(mean_squared_error(y_teste, y_pred_v1)),3))
    print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(mean_squared_error(y_teste, y_pred_v1))),3))
    print('R2 Score:', round(r2_score(y_teste, y_pred_v1),6))

def _training_model_decision_tree(X_treino_scaled: pd.DataFrame, y_treino: pd.DataFrame, X_teste_scaled: pd.DataFrame, y_teste: pd.DataFrame):  
    print("_training_model_decision_tree()")

    model = DecisionTreeRegressor(random_state = 1)
    model.fit(X_treino_scaled, y_treino)

    print(model.get_params())

    y_pred_v1 = model.predict(X_teste_scaled)

    # Métricas
    print('Mean Absolute Error (MAE):', round(mean_absolute_error(y_teste, y_pred_v1),3))  
    print('Root Mean Squared Error (RMSE):', round(np.sqrt(mean_squared_error(y_teste, y_pred_v1)),3))
    print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(mean_squared_error(y_teste, y_pred_v1))),3))
    print('R2 Score:', round(r2_score(y_teste, y_pred_v1),6))

def _training_model_random_forest(X_treino_scaled: pd.DataFrame, y_treino: pd.DataFrame, X_teste_scaled: pd.DataFrame, y_teste: pd.DataFrame):  
    print("_training_model_random_forest()")

    model = RandomForestRegressor(random_state = 1)
    model.fit(X_treino_scaled, y_treino)

    print(model.get_params())

    y_pred_v1 = model.predict(X_teste_scaled)

    # Métricas
    print('Mean Absolute Error (MAE):', round(mean_absolute_error(y_teste, y_pred_v1),3))  
    print('Root Mean Squared Error (RMSE):', round(np.sqrt(mean_squared_error(y_teste, y_pred_v1)),3))
    print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(mean_squared_error(y_teste, y_pred_v1))),3))
    print('R2 Score:', round(r2_score(y_teste, y_pred_v1),6))