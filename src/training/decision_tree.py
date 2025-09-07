import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def training_model_decision_tree(X_treino_scaled: pd.DataFrame, y_treino: pd.DataFrame, X_teste_scaled: pd.DataFrame, y_teste: pd.DataFrame):  
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
