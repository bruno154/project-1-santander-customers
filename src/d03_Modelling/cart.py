import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from yellowbrick.model_selection import ValidationCurve


def random_cart(X_treino, y_treino, n_iter, model_name):

    """

    :param X_treino: Variáveis independentes de treino.
    :param y_treino: Variável dependente de treino.
    :param n_iter: numero de iterações.
    :param model_name: string com novo do arquivo pickle.
    :return: O best_estimator_, cv_results_, best_params_
    """

    print('> Procurando os melhores parametros...')
    splitter = ['best', 'random']
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 110, num=7)]
    max_depth.append(None)
    criterion = ['gini', 'entropy']
    min_samples_split = [2, 5, 10, 12]
    min_samples_leaf = [1, 2, 4, 6, 8]

    random_grid = {'splitter': splitter,
                   'max_features': max_features,
                   'criterion': criterion,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    random_state = 2
    cart = DecisionTreeClassifier()
    cart_random = RandomizedSearchCV(estimator=cart, param_distributions=random_grid,
                                     n_iter=n_iter, cv=3, verbose=1, random_state=random_state,
                                     n_jobs=-1, scoring={'f1': 'f1'},
                                     refit='f1')

 
    print('> Treinando Modelo...')
    model = cart_random.fit(X_treino, y_treino)
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

    viz = ValidationCurve(
        model, param_name=param,
        param_range=rang, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    viz.fit(X, y)
    viz.show()

def fitting_model_cart(X, y, model_name, splitter='best' , max_features='sqrt', max_depth=11 , 
                       criterion='entropy', min_samples_split=2, min_samples_leaf=2, 
                       random_state=2):

    """
    
    :param X:                   Variáveis de treino
    :param y:                   Variável target de treino
    :param model_name:          Nome do arquivo .sav onde o modelo sera salvo - str
    :param demais parametros:   Iguais aos do Sklearn
    :return:                    Modelo treinado
    """
    
    cart = DecisionTreeClassifier(splitter=splitter, max_features=max_features, max_depth=max_depth, criterion=criterion, 
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                  random_state=random_state)
    
    
    # Fit do modelo
    print('> Treinando Modelo...')
    model = cart.fit(X, y)
    filename = model_name
    pickle.dump(model, open(filename, 'wb'))
    print('>Modelo Treinado...')
    return model
