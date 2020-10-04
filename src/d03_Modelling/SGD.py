import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from yellowbrick.model_selection import ValidationCurve


def random_SGD(X_treino,y_treino,n_iter, model_name):

    """

    :param X_treino: Variáveis independentes de treino.
    :param y_treino: Variável dependente de treino.
    :param n_iter: numero de iterações.
    :param model_name: string com novo do arquivo pickle.
    :return: O best_estimator_, cv_results_, best_params_
    """
    
    print('> Procurando os melhores parametros...')
    loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    penalty = ['l2', 'l1','elasticnet']
    alpha = [0.000050] #[float(x) for x in np.linspace(0.00001, 0.001, num=7)]
    max_iter = [int(x) for x in np.linspace(500, 1500, num=5)]
    learning_rate = ['optimal']
   
    

    random_grid = {'penalty' : penalty,
                   'alpha':alpha,
                   'max_iter': max_iter,
                   'learning_rate': learning_rate}
    

    random_state = 2
    sgd = SGDClassifier(early_stopping=True)
    sgd_random = RandomizedSearchCV(estimator = sgd, param_distributions = random_grid, 
                                   n_iter = n_iter, cv = 3, verbose=1, random_state=random_state, 
                                   n_jobs = -1, scoring = {'roc_auc':'roc_auc'}, 
                                   refit='roc_auc')

    # Fit do modelo
    print('> Fitting Modelo...')
    model = sgd_random.fit(X_treino, y_treino)
    filename = model_name
    pickle.dump(model.best_estimator_, open(filename, 'wb'))
    
    print('> Treinamento realizado...')
    return model.best_estimator_, model.cv_results_, model.best_params_



def validation_curve(model, X, y, param, rang, cv):

    """

    :param model: Modelo a ser avaliado.
    :param x: Variáveis independentes de treino.
    :param y: Variavel dependente de treino.
    :param param: Parametro do modelo a ser avaliado.
    :param rang: Espaço de hipotese do parametro que esta sendo avaliado.
    :param cv: quantidade de splits para a cross validação.
    :return: Viz das curvas de validação.
    """
    
    f, ax  = plt.subplots(figsize=(10,6))
    viz = ValidationCurve(
    model, param_name=param,
    param_range=rang, cv=cv, scoring="roc_auc", n_jobs=-1
    )


    viz.fit(X, y)
    viz.show()
    plt.show()



def fitting_model_sgd(X, y, model_name, penalty ='l1', alpha = 0.000050,
    max_iter = 500, learning_rate ='optimal'):

    """
    
    :param X:                   Variáveis de treino
    :param y:                   Variável target de treino
    :param model_name:          Nome do arquivo .sav onde o modelo sera salvo - str
    :param demais parametros:   Iguais aos do Sklearn
    :return:                    Modelo treinado
    """
    
    
    sgd = SGDClassifier( penalty=penalty, alpha= alpha,
    max_iter=max_iter, learning_rate=learning_rate, early_stopping=True)

    
    # Fit do modelo
    print('> Treinando Modelo...')
    model = sgd.fit(X, y)
    filename = model_name
    pickle.dump(model, open(filename, 'wb'))
    print('> Modelo Treinado...')
    return model

    