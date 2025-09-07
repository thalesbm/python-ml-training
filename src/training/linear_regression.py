import pandas as pd
import numpy as np

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset.training_dataset import create_training_dataset

class LinearRegressionAlgorithm:

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def training_model(self):  
        print("training_model()")

        X_treino, X_teste, y_treino, y_teste = create_training_dataset(dataset=self.dataset)
        X_treino_scaled, X_teste_scaled = self._padronize_dataset(X_treino=X_treino, X_teste=X_teste)

        model = LinearRegression()
        model.fit(X_treino_scaled, y_treino)

        y_pred_v1 = model.predict(X_teste_scaled)

        print("--------------------------------")
        print("Metricas Linear Regression:")
        print("Parametros: ", model.get_params())
        print('Mean Absolute Error (MAE):', round(mean_absolute_error(y_teste, y_pred_v1),3))  
        print('Root Mean Squared Error (RMSE):', round(np.sqrt(mean_squared_error(y_teste, y_pred_v1)),3))
        print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(mean_squared_error(y_teste, y_pred_v1))),3))
        print('R2 Score:', round(r2_score(y_teste, y_pred_v1),6))
        print("--------------------------------")

    def _padronize_dataset(self, X_treino: pd.DataFrame, X_teste: pd.DataFrame):
        print("_padronize_dataset()")

        scaler = StandardScaler()
        X_treino_scaled = scaler.fit_transform(X_treino)
        X_teste_scaled = scaler.transform(X_teste)
        pickle.dump(scaler, open('model/dsa_scaler_linear_regression.pkl','wb'))

        return X_treino_scaled, X_teste_scaled
