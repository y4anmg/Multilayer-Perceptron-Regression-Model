import preprocessingB as pre
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

#Hiperparametros do modelo
hls = (10, 5)
alpha = 0.1
activation = 'tanh'
Solver = 'sgd'
filename = "data_filtered_importanceRF.csv"
random_state = 1
max_iter = 1000

def computeMLPRegressor(X, y, random_state, max_iter, activation, alpha, hidden_layer_sizes, solver):
    X, y = pre.loadDataset(filename)
    X, y = pre.computeScaling(X, y)
    regressorMLP = MLPRegressor(random_state=random_state, max_iter=max_iter, activation=activation, alpha=alpha, hidden_layer_sizes=hls)
    regressorMLP.fit(X, y)
    return regressorMLP

def runMLPRegressor(filename, random_state, max_iter):
    X, y = pre.loadDataset(filename)
    X, y = pre.computeScaling(X, y.ravel())
    #Remova para randomizar y
    #indices_random = list(range(len(y)))
    #random.shuffle(indices_random)
    #y = y[indices_random]
    regressorMLP = computeMLPRegressor(X, y, random_state=random_state, max_iter=max_iter, activation=activation, alpha=alpha, hidden_layer_sizes=hls, solver=Solver)
       
    r2 = r2_score(y, regressorMLP.predict(X))
    print(r2)
    y_pred = regressorMLP.predict(X) 
    
    # Plotando o gráfico de dispersão
    plt.scatter(y, y_pred, c='blue', label='Valores Previstos')
    plt.scatter(y, y, c='red', label='Valores Reais')
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    plt.title("Gráfico de Dispersão: Valores Reais vs Valores Previstos")
    plt.savefig("y_ypred.png")
    #plt.show()
    return r2, y, y_pred

def evaluateMetrics(y, y_pred):
    from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error

# Calcule as métricas de desempenho
    variance_explained = explained_variance_score(y, y_pred)
    max_residual_error = max_error(y, y_pred)
    mean_abs_error = mean_absolute_error(y, y_pred)
    mean_squared_err = mean_squared_error(y, y_pred)
    
# Crie um dicionário com as métricas
    metrics_dict = {
        'Métrica': ['R^2', 'Variancia Explicada', 'Erro Residual Máximo', 'Erro Médio Absoluto', 'Erro Quadrático Médio'],
        'Valor': [r2, variance_explained, max_residual_error, mean_abs_error, mean_squared_err]
}

# Crie um DataFrame do Pandas com as métricas
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv('metricas.csv')
# Imprima a tabela
    print(metrics_df)

def calculate_leverage_and_plot(X, y, regressorMLP):
   # Parâmetros:
    X, y = pre.loadDataset(filename)
    X, y = pre.computeScaling(X, y.ravel())
    regressorMLP = computeMLPRegressor(X, y, random_state=random_state, max_iter=max_iter, activation=activation, alpha=alpha, hidden_layer_sizes=hls, solver=Solver)
    def calculate_leverage(X_train, regressorMLP):
        y_pred = regressorMLP.predict(X_train)  # Prediz os valores do conjunto de treinamento
        residuos = y - y_pred  # Calcula os resíduos
        desvio_padrao_residuos = np.sqrt(np.mean(residuos ** 2))  # Calcula o desvio padrão dos resíduos
        residuos_padronizados = residuos / desvio_padrao_residuos  # Calcula os resíduos padronizados
        H = np.dot(X_train, np.dot(np.linalg.pinv(np.dot(X_train.T, X_train)), X_train.T))  # Calcula a matriz de projeção
        leverage_values = np.diagonal(H)  # Calcula os valores de alavancagem para todas as amostras
        return leverage_values

    # Função para calcular o limite máximo de alavancagem
    def calculate_leverage_threshold(p, n):
        leverage_threshold = (p / n) * 3  # Calcula o limite máximo de alavancagem
        return leverage_threshold

    leverage_values = calculate_leverage(X, regressorMLP)
    p = X.shape[1]  # Número de características
    n = X.shape[0]  # Número de amostras no conjunto de treinamento
    leverage_threshold = calculate_leverage_threshold(p, n)
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(leverage_values)), leverage_values, label='Amostras', color='blue')

        # Gerar gráfico
    #plt.figure(figsize=(8, 6))
    #plt.scatter(range(len(leverage_values)), leverage_values, label='Resíduos Padronizados', color='blue')
    plt.axhline(y=leverage_threshold, color='red', linestyle='--', label='Limite Leverage')
    plt.xlabel('Resíduos Padronizados')
    plt.ylabel('Pontuação Leverage')
    plt.title('Pontuação Leverage vs Resíduos Padronizados')
    plt.legend()
    plt.savefig("filteredbyRF_leverage_score.png")
    plt.show()
    
    df = pd.DataFrame({'Original_Index': np.arange(len(leverage_values)), 'Leverage': leverage_values})
    df_sorted = df.sort_values(by='Leverage', ascending=True)
    df_sorted.to_csv('sorted_Leverage.csv', index=False)


# Exemplo de uso:
X, y = pre.loadDataset(filename)
X, y = pre.computeScaling(X, y.ravel())
regressorMLP = computeMLPRegressor(X, y, random_state=random_state, max_iter=max_iter, activation=activation, alpha=alpha, hidden_layer_sizes=hls, solver=Solver)
calculate_leverage_and_plot(X, y, regressorMLP)
r2, y, y_pred = runMLPRegressor(filename, random_state, max_iter)
evaluateMetrics(y, y_pred)
runMLPRegressor(filename, random_state, max_iter)
leverageScore(MLPRegressor)
