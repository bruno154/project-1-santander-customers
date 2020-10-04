import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_balance(dados,target):
    vals = dados[target].unique()
    print(f'As classes do dataset são: {vals}')
    print('\n')
    sns.set(style = 'darkgrid')
    f, (ax) = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target, data = dados, ax= ax)