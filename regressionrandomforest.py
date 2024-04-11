import preprocessingB as pre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

def computeRandomForestRegressionModel(X, y, numberOfTrees, criterion='entropy'):
    from sklearn.ensemble import RandomForestRegressor
    
    regressor = RandomForestRegressor(n_estimators = numberOfTrees)
    regressor.fit(X, y)

    return regressor

def plot_feature_importance(regressor, X):
    X, y = pre.loadDataset("data_no_fukui.csv")                                                    
    regressor = computeRandomForestRegressionModel(X, y, 100)
    
   
    importances = regressor.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
    X_names = pd.read_csv("data_no_fukui.csv")
    X_names = X_names.drop(['SMILES', 'pIC50'], axis=1)
    feature_names = X_names.columns.to_list()
    #feature_names = [f"feature {i}" for i in range(X.shape[1])]
    # Criar uma série pandas com as importâncias das features
    forest_importances = pd.Series(importances, index=feature_names, name='Importance')
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    sorted_importance = forest_importances.sort_values(ascending=False)
    sorted_importance.to_csv('sorted_importance.csv', header=['Importance'])
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature {i}" for i in range(X.shape[1])])
    half_len = len(X.columns) // 2
    first_half = list(range(half_len))
    second_half = list(range(half_len, len(X.columns)))
   #Plotar os gráficos
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 subgráficos em uma coluna
    forest_importances.iloc[first_half].plot.bar(yerr=std[first_half], ax=axs[0])
    axs[0].set_title("Feature importances (Parte 1)")
    axs[0].set_ylabel("Mean decrease in impurity")

    forest_importances.iloc[second_half].plot.bar(yerr=std[second_half], ax=axs[1])
    axs[1].set_title("Feature importances (Parte 2)")
    axs[1].set_ylabel("Mean decrease in impurity")
    #x.set_title("Feature importances using MDI (Random Forest Regression)")
    #x.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('feature_importance_entropy.png')
    plt.show()

def calculate_permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=2):
    result = permutation_importance(regressor, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs)

    feature_importances = result.importances_mean
    feature_std = result.importances_std

    return feature_importances, feature_std

def calculatePermutationImportance(model, X, y):
    X, y = pre.loadDataset("data_no_fukui.csv") 
    X, y = pre.splitTrainTestSets(X, y, 0.2)
    regressor = computeRandomForestRegressionModel(X, y, 100)
    result = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

def showPlot(XPoints, yPoints, XLine, yLine):
    X, y = pre.loadDataset("data_no_fukui.csv")
    regressor = computeRandomForestRegressionModel(X, y, 100)
    # Obtendo os valores previstos pelo modelo
    y_predito = regressor.predict(X)
    for i in range(X.shape[1]):
      plt.scatter(X[:, i], y, color='red', label=f'Valores Reais - Conjunto {i+1}')
      plt.scatter(X[:, i], y_predito, color='blue', label=f'Valores Preditos - Conjunto {i+1}', alpha=0.5)
    r2 = r2_score(y, regressor.predict(X))
    print(r2)
    plt.title("Regressão de floresta aleatória, R2 = {:.3f}".format(r2))
    plt.xlabel("Descritores")
    plt.ylabel("IC50")
    plt.legend()
    plt.savefig('random_forest_100trees_pIC50.png')
    plt.show()

def runRandomForestRegressionExample(filename, numberOfTrees):
    X, y = pre.loadDataset(filename)
    X, y = pre.computeScaling(X, y)
    regressor = computeRandomForestRegressionModel(X, y, 100)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, regressor.predict(X))
    print("Coeficiente de Determinação (R2): {:.3f}".format(r2))

    return r2    

if __name__ == "__main__":
    regressor, X, y = runRandomForestRegressionExample("data_no_fukui.csv", 100)
    plot_feature_importance(regressor, X)
    calculatePermutationImportance(regressor, X, y)
