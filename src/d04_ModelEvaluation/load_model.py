import os
import pickle
import pandas as pd

def load_model(model):
    os.chdir(input("Qual o diretorio do modelo?"))
    file = open(model, 'rb')
    model = pickle.load(file)
    return model



