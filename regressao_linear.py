# coder: Debora Azevedo Caetano

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats

def main():
    dados = pd.read_csv('Trabalho.csv')
    X = dados[['X1', 'X2', 'X3', 'X4']]
    y = dados['Y']
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

    modelos = {
        'Modelo 1 - KNN (K=1)': KNeighborsRegressor(n_neighbors=1),
        'Modelo 2 - KNN (K=3)': KNeighborsRegressor(n_neighbors=3),
        'Modelo 3 - KNN (K=5)': KNeighborsRegressor(n_neighbors=5),
        'Modelo 4 - Regressão_Linear': LinearRegression()
    }

    erros = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        y_predito = modelo.predict(X_teste)
        erro = np.abs(y_teste - y_predito)
        erros[nome] = erro


    mae_resultados = {}
    for nome, erro in erros.items():
        mae = np.mean(erro)
        mae_resultados[nome] = mae

    confianca = 0.95
    n = len(y_teste)
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    intervalos = {}

    for nome, erro in erros.items():
        desvio_padrao = np.std(erro)
        erro_padrao = desvio_padrao / np.sqrt(n)
        intervalo_inferior = mae_resultados[nome] - z * erro_padrao
        intervalo_superior = mae_resultados[nome] + z * erro_padrao
        intervalos[nome] = (intervalo_inferior, intervalo_superior)


    print("\n----- Erro Absoluto Médio (MAE) -----")
    for nome, mae in mae_resultados.items():
        print(f'{nome} --> |{mae:.2f}|')

    print("\n\n----- Intervalos de Confiança (95%) -----")
    for nome, intervalo in intervalos.items():
        print(f'{nome} --> ({intervalo[0]:.2f}, {intervalo[1]:.2f})')

    print("\n")

if __name__ == "__main__":
    main()
