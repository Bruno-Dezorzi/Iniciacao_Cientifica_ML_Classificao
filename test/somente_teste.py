import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.concat([load_iris(return_X_y= True, as_frame= True)[0],load_iris(return_X_y= True, as_frame= True)[1]], axis= 1)
df.head()
print(f"{df.dtypes}\n")
print(df.info())
display(df.describe())
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

df['species'] = df['target'].map(species_mapping)
df = df.drop('target', axis = 1)
df.head()
df_numeric = df.drop('species', axis=1)
corr = df_numeric.corr()


plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)


plt.tight_layout()
plt.show()

sns.pairplot(df, hue="species")  # Substitua "target" pela variável alvo
plt.show()
def boxplot(df: pd.DataFrame):
    df_numeric = df.drop('species', axis=1)

    plt.figure(figsize = (20,10))

    sns.boxplot(data=df_numeric)
    plt.show()

def histogram(df: pd.DataFrame):
    columns_numerics = df.drop('species', axis=1).columns.tolist()
    
    plt.figure(figsize= (20,10))
    for i in range(len(columns_numerics)):
        plt.subplot(4,4,i+1)
        sns.histplot(data= df, x= columns_numerics[i], kde= True)
        #plt.title(columns_numerics[i])

    plt.tight_layout()
    plt.show()
    
boxplot(df)
histogram(df)
def detectar_outliers(df: pd.DataFrame, colunas: list, info: bool = False):
    all_outliers = {}
    
    for i in colunas:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Identificando os outliers
        outliers = df[(df[i] < limite_inferior) | (df[i] > limite_superior)]

        all_outliers[i] = outliers
        
        if info:
            print(f"Sobre a coluna {i}: O Limite inferior é {limite_inferior} e o superior é {limite_superior}")
            print(f"{len(outliers)} outliers na coluna {i}\n")
            
    return all_outliers

def remover_outliers(df: pd.DataFrame, colunas: list):
    for i in colunas:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Detecção de outliers
        outliers = detectar_outliers(df, [i], info=False)[i]
        
        # Calculando a média dos valores sem os outliers
        media_no_outliers = df[(df[i] >= limite_inferior) & (df[i] <= limite_superior)][i].mean()
        
        # Substituindo os outliers pela média
        df[i] = df[i].apply(
            lambda x: media_no_outliers if (x < limite_inferior or x > limite_superior) else x
        )
    return df
df = remover_outliers(df= df, colunas=df.drop('species', axis=1).columns.tolist())
boxplot(df)
histogram(df)
df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
df["petal_prop"] = df["petal length (cm)"] / df["petal width (cm)"]
df["sepal_prop"] = df["sepal length (cm)"] / df["sepal width (cm)"]
df.describe()
boxplot(df)
histogram(df)
display(df.describe())
for i in range(5):
    df = remover_outliers(df= df, colunas=df.drop('species', axis=1).columns.tolist())
    i += 1
boxplot(df)
histogram(df)
plt.figure(figsize= (15,7))
sns.scatterplot(data=df.drop('species', axis=1))

plt.tight_layout()
plt.show()
plt.figure(figsize= (15,7))
sns.scatterplot(data=df.drop('species', axis=1))

plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(df.drop('species', axis=1))
y = df['species']


plt.figure(figsize= (15,7))
sns.scatterplot(data=X)

plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 8))

for i in range(X.shape[1]):
    plt.subplot(2, 4, i+1)
    sns.histplot(X[:, i], bins=10, edgecolor='black', kde = True)
    plt.title(f'Histograma da Coluna {i+1}')

plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split

seed = 42


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score



dummy_clf = DummyClassifier(strategy="most_frequent", random_state=seed)

dummy_clf.fit(X_train, y_train)


y_test_pred = dummy_clf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia no conjunto de teste: {test_accuracy:.2f}')
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

kernels = ['linear', 'poly', 'rbf', 'sigmoid'] #'precomputed' não foi utilizado pois precisa de uma matriz quadrada

for i in kernels:
    svm = SVC(kernel=i, random_state= 42)
    
    # Cross-validation com 5 folds
    scores = cross_val_score(svm, X_train, y_train, cv=5)
    
    print(f'Kernel: {i}')
    print(f'Acurácia média no Cross-Validation: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})\n')

# Após escolher o melhor kernel, podemos treinar e testar no conjunto final
best_kernel = 'linear'  # Substitua pelo melhor kernel encontrado
svm_best = SVC(kernel=best_kernel)
svm_best.fit(X_train, y_train)




y_test_pred = svm_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia final no conjunto de teste para kernel {best_kernel}: {test_accuracy:.2f}')

from sklearn.naive_bayes import GaussianNB, BernoulliNB


methods = [GaussianNB, BernoulliNB]  # Apenas métodos que aceitam valores negativos   / MultinomialNB, CategoricalNB e ComplementNB não aceita valores negativos

for method in methods:
    naive = method()

    # Cross-validation com 5 folds
    scores = cross_val_score(naive, X_train, y_train, cv=5)

    print(f'Método: {method.__name__}')
    print(f'Acurácia média no Cross-Validation: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})\n')


best_method = GaussianNB  # Substitua pelo melhor encontrado
naive_best = best_method()
naive_best.fit(X_train, y_train)





y_test_pred = naive_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia final no conjunto de teste para {best_method.__name__}: {test_accuracy:.2f}')

best_method = BernoulliNB  # Substitua pelo melhor encontrado
naive_best = best_method()
naive_best.fit(X_train, y_train)





y_test_pred = naive_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia final no conjunto de teste para {best_method.__name__}: {test_accuracy:.2f}')

svm = SVC(kernel='linear', C= 0.1, random_state= seed, class_weight='balanced')
svm.fit(X_train, y_train)



y_test_pred = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia final no conjunto de teste para kernel {best_kernel}: {test_accuracy:.6f}')

gaussian = GaussianNB(var_smoothing= 1e-1)  
gaussian.fit(X_train, y_train)





y_test_pred = gaussian.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Acurácia final no conjunto de teste para GaussianNB: {test_accuracy:.6f}')