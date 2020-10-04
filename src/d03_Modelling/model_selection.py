#Importações

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def model_selection(x_treino, y_treino,x_teste,y_teste, num_folds = 10):

    """

    :params models:      Modelos a serem testados
    :params x_treino:    Dados de Treino variaveis independentes
    :params y_treino:    Dados de Treino variavel dependente
    :params x_teste:     Dados de Teste variaveis independentes
    :params y_teste:     Dados de Teste variaveis dependente
    :params num_folds:   Número de vezes que os dados seram divididos
    :return:             Retorna os indices ROCAUC dos modelos assim como seus desvios-padrões    
    """
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SGD', SGDClassifier()))


    resultados = []
    nomes = []
    resultados_teste = []
    nomes_teste = []
    
    
    print('Dados de treino...')
    for nome, modelo in models:
        kfold = KFold(n_splits = num_folds, random_state = 42)
        cv_results = cross_val_score(modelo, x_treino, y_treino, cv = kfold, scoring = 'roc_auc')
        resultados.append(cv_results)
        nomes.append(nome)
        print(f'{nome}- ROCAUC: {cv_results.mean()} Std: {cv_results.std()}')
    
    print('\n')
    
    print('Dados de teste...')
    for nome, modelo in models:
        kfold = KFold(n_splits = num_folds, random_state = 24)
        cv_results_teste = cross_val_score(modelo, x_teste, y_teste, cv = kfold, scoring = 'roc_auc')
        resultados_teste.append(cv_results_teste)
        nomes_teste.append(nome)
        print(f'{nome}- ROCAUC: {cv_results_teste.mean()} Std: {cv_results_teste.std()}')
    return models


def learning_curves(models, X, y):

    """
    :params models:  Modelos a serem avaliados
    :params X:       Dados de Treino variaveis independentes
    :params y:       Dados de Treino variavel dependente
    :return:         Viz da curvas de apendizagem
    """
    
    cv_strategy = StratifiedKFold(n_splits=3)
    for model in models:
        
        sizes = np.linspace(0.3,1.0,10)
        viz = LearningCurve(model, cv=cv_strategy, scoring='roc_auc', train_sizes=sizes, n_jobs=4)
        viz.fit(X, y)
        viz.show()
