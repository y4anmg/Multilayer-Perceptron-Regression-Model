import numpy as np
import pandas as pd

#Cria um objeto chamado baseDeDados que recebe valores através do pandas que usa o modulo read_csv, recebdno dois parametros (filename e delimiter)

def loadDataset(filename):
    print('Carregando base de dados...')
    baseDeDados = pd.read_csv(filename, delimiter=',')
    X = baseDeDados.iloc[:,1:-1].values #X recebe todas as linhas do DF e todas as colunas menos a última, especificando a seleção apenas dos valores.
    y = baseDeDados.iloc[:,-1].values #Apenas a ultima coluna]
    print('Base de dados carregada!')
    return X, y

#Muda palavras do conjunto de dados para números aleatrórios
def computeCategorization(X):
     print('Computando rótulos...')
     from sklearn.preprocessing import LabelEncoder
     labelencoder_X = LabelEncoder() #Cria o objeto transformador
     X[:,0] = labelencoder_X.fit_transform(X[:,0])
    # D = pd.get_dummies(X[:, 0], prefix='SMILES', drop_first=True) #Nesse modelo não será utilizado o nome dos SMILES
    # X = np.column_stack((D, X[:, 1:]))
     return X
    
#Separa dados em teste/treino
def splitTrainTestSets(X, y, test_size):
    print('Separando dados em teste/treino...')
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2) #test_size parametrizado em 20%
    return XTrain, XTest, yTrain, yTest

def computeScaling(train, test):
    print('Normalizando dados...')
    # Verifica se train tem apenas uma dimensão (1D)
    if train.ndim == 1:
        train = train.reshape(-1, 1)

    # Verifica se test tem apenas uma dimensão (1D)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    test = scaleX.fit_transform(test)
    train = np.round(train, 4)
    test = np.round(test, 4)
    print("Dados normalizados!")
    return train, test


       
