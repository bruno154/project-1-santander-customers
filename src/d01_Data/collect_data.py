import os 
import pandas as pd


def collect_data(file_treino, file_teste):

    """
    :param file_treino: Arquivo com os dados de treino. str, .csv:
    :param file_teste:  Arquivo com os dados de teste. str, .csv:
    :param var_target:  Nome da variável Target. str:
    :return: Pandas Dataframe das variáveis X e da variável y e dos dados de treino e teste.

    """
    path = input('Por favor adicione o diretório onde estão os dados?')
    for dirname, _, filename in os.walk(path, topdown = True):
        for filename in filename:
            if filename == file_teste:
                    teste = pd.read_csv(os.path.join(dirname,filename))
            else:
                if filename == file_treino:
                    treino = pd.read_csv(os.path.join(dirname,filename))
                else:
                    print("Carregamento não foi possível")


    X = treino.drop('TARGET',axis=1) 
    y= treino['TARGET']

    return X , y, treino, teste


def load_data(file):
    """
    :param file: Arquivo a ser carregado. str, .csv: 
    :return: Dataframe do Arquivo carregado.
    """
    path = input('Por favor adicione o diretório de trabalho?')
    for dirname, _, filename in os.walk(path, topdown=True):
        for filename in filename:
            if filename == file:
                data = pd.read_csv(os.path.join(dirname, filename), header=None)
            else:
                pass
    print("Carregamento finalizado!!!")
    return data
