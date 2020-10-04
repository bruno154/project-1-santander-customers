import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.model_selection import LearningCurve
from yellowbrick.classifier import ROCAUC, ConfusionMatrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


def class_report(model, X, Y, x, y, cv=10, scoring='roc_auc'):
    
    cv_strategy = StratifiedKFold(n_splits=cv)
    
    pred = model.predict(x)
    print(classification_report(y,pred)) 
    print(f"A ROC_AUC cross-validada nos dados de treino é {round(np.mean(cross_val_score(model, X, Y,scoring=scoring, cv=cv_strategy, n_jobs=-1)),2)}")
    return print(f"A ROC_AUC cross-validada nos dados de teste é {round(np.mean(cross_val_score(model, x, y,scoring=scoring, cv=cv_strategy, n_jobs=-1)),2)}")

def evaluation(estimator, X, Y, x, y):
    
    classes = [Y[1],Y[0]]
    f, (ax,ax1,ax2) = plt.subplots(1,3,figsize=(18,6))

    #Confusion Matrix
    cmm = ConfusionMatrix(model=estimator,ax=ax1,  classes=classes, label_encoder={0.0: 'Negativo', 1.0: 'Positivo'})
    cmm.score(x, y)

    #ROCAUC
    viz = ROCAUC(model=estimator, ax=ax2)
    viz.fit(X,Y)
    viz.score(x,y)

    #Learning Curve
    cv_strategy = StratifiedKFold(n_splits=3)
    sizes = np.linspace(0.3,1.0,10)
    visualizer = LearningCurve(estimator, ax=ax, cv=cv_strategy, scoring='roc_auc', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X, Y)        


    cmm.poof(), viz.poof(), visualizer.poof()
    plt.show()
    