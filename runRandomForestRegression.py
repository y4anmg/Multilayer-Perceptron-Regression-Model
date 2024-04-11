import preprocessingB as pre
import regressionrandomforest as rf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def runRandomForestRegressionExample(filename):   
    X, y = pre.loadDataset("data_no_fukui.csv")
    rfModel = rf.computeRandomForestRegressionModel(X, y.ravel(), 100)
    #rf.showPlot(X, y, X, rfModel.predict(X))
    rf.plot_feature_importance(rfModel, X)
# Suponha que 'target_column' seja a coluna que você deseja prever
# Chame a função para calcular a importância de características baseada em permutação
    #rf.calculatePermutationImportance(rfModel, X, y)

runRandomForestRegressionExample("data_no_fukui.csv")
