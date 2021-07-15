from sklearn.preprocessing import StandardScaler
import pandas as pd
# import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn import model_selection as ms
from regressao_knn import RegressaoKnn

def real_brazilian_to_float(numeric_string):
    """Converte valores numérios no formato brasileiro 1.234,56 para float"""
    try:
        return float(numeric_string.replace(".", "").replace(",", "."))
    except:
        return numeric_string


class ConsumoCerveja:

    def __init__(self):

        df = pd.read_csv("archive/Consumo_cerveja.csv").dropna()

        # Conversão para float

        colunas_numericas = ['Temperatura Media (C)',
                             'Temperatura Minima (C)',
                             'Temperatura Maxima (C)',
                             'Precipitacao (mm)']

        df[colunas_numericas] = df[colunas_numericas].applymap(
            real_brazilian_to_float)

        # %% separação dos dados

        X = df[['Temperatura Media (C)',
                'Temperatura Minima (C)',
                'Temperatura Maxima (C)',
                'Precipitacao (mm)',
                'Final de Semana']]

        y = df[['Consumo de cerveja (litros)']]

        # %% separar treino e teste

        splits = ms.train_test_split(X, y, test_size=0.2)
        X_train, X_test, y_train, y_test = splits

        # %% Normalização Z-score

        # Instanciar objeto para normalização Z-score
        scaler = StandardScaler()

        scaler_previsores = scaler.fit(X_train)
        X_train = scaler_previsores.transform(X_train)
        X_test = scaler_previsores.transform(X_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def estimar_qualidade(self, n_neighbors):

        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        # model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model = RegressaoKnn(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)

        expected = y_test
        predicted = model.predict(X_test)

        return {"MSE": metrics.mean_squared_error(expected, predicted),
                "R2": metrics.r2_score(expected, predicted)}
