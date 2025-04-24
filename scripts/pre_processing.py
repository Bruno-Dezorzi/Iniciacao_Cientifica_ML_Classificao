# Importações
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# Carregamento do dataset Iris
iris = load_iris(as_frame=True)
df = pd.concat([iris.data, iris.target.rename('target')], axis=1)

# Mapeamento de classes numéricas para nomes de espécies
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['target'].map(species_mapping)
df.drop(columns='target', inplace=True)

# Funções para detecção e tratamento de outliers
def detectar_outliers(df: pd.DataFrame, colunas: list, info: bool = False):
    all_outliers = {}
    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)]
        all_outliers[col] = outliers
        if info:
            print(f"Coluna {col}: Limite inferior = {lim_inf}, superior = {lim_sup}")
            print(f"{len(outliers)} outliers encontrados\n")
    return all_outliers

def remover_outliers(df: pd.DataFrame, colunas: list):
    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        media = df[(df[col] >= lim_inf) & (df[col] <= lim_sup)][col].mean()
        df[col] = df[col].apply(lambda x: media if (x < lim_inf or x > lim_sup) else x)
    return df

# Remoção de outliers
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df = remover_outliers(df, num_cols)

# Feature Engineering

df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
df['petal_prop'] = df['petal length (cm)'] / df['petal width (cm)']
df['sepal_prop'] = df['sepal length (cm)'] / df['sepal width (cm)']

# Salvando o DataFrame processado com pickle
with open('data/iris_processado.pkl', 'wb') as f:
    pickle.dump(df, f)

print("✅ DataFrame processado salvo como 'iris_processado.pkl'.")