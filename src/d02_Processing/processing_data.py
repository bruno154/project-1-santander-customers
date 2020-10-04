import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def processing_data(X, y):
    """
    Função de processamento dos dados para posterior
    treinamento do Modelo de Machine Learning

      1. Balanciar Dataset
      2. Aplicar MinMaxScaler
      3. Aplicar a redução de dimensionalidade - PCA

    """

    # Balanciando o dataset
    x, y = SMOTE().fit_resample(X, y)
    print('Balanciamento realizado: ', sorted(Counter(y).items()))

    # Normalização
    scl = MinMaxScaler()
    scl_x = scl.fit_transform(x)
    print("Dados Normalizados")

    # Redução da Dimensionalidade
    pca = PCA(n_components=50)
    x_fit = pca.fit(scl_x)
    x_red = pca.fit_transform(scl_x)
    print("Redução de Dimensionalidade concluída")
    print(f'Variância capturada de {round(x_fit.explained_variance_ratio_.sum() * 100, 2)}')

    # Variáveis de Treino e Teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(x_red, y, test_size=0.2, random_state=42)
    print("Divisão entre dados de treino e teste concluída")

    return X_treino, X_teste, y_treino, y_teste


def save_processed_data(filesname, files):
    """
    Função para utilitária para salvar array numpy como .csv

    Parametros:
    filesname: lista de string com o nome do arquivo ex. 'arquivo.csv'
    files: lista de arrays numpy

    """

    path = input('Por favor adicione o diretório de trabalho?')
    os.chdir(path)
    for a, b in zip(filesname, files):
        np.savetxt(a, b, delimiter=',', encoding='utf-8')
    return print("Arquivos salvos")


def load_processed_data(X_treino, y_treino, X_teste, y_teste):

    """
    :param X_treino: Arquivo com os dados de treino, variaveis independentes. str, .csv:
    :param y_treino: Arquivo com os dados de treino, variavel dependente. str, .csv
    :param X_teste:  Arquivo com os dados de teste, variaveis independentes. str, .csv: 
    :param y_teste:  Arquivo com os dados de teste, variavel dependente. str, .csv
    :return: Pandas Dataframe das variáveis X e da variável y divididos em dados de treino e teste.

    """
    path = input('Por favor adicione o diretório onde estão os dados?')
    for dirname, _, filename in os.walk(path, topdown = True):
        for filename in filename:

            if filename == X_treino:
                X = pd.read_csv(os.path.join(dirname,filename),header=None)
            elif filename == y_treino:
                Y = pd.read_csv(os.path.join(dirname,filename),header=None)
            elif filename == X_teste:
                x = pd.read_csv(os.path.join(dirname,filename),header=None)
            elif filename == y_teste:
                y = pd.read_csv(os.path.join(dirname,filename),header=None)
            else:
                pass
        print("Carregamento Finalizado!!!")


    return X,Y,x,y

  