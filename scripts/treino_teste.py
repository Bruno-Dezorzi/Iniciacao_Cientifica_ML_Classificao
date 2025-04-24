import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB

with open('data/iris_processado.pkl', 'rb') as f:
    df = pickle.load(f)

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('species', axis=1))
y = df['species']

# Divisão dos dados em treino, validação e teste
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seed)

# DummyClassifier (baseline)
dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
dummy.fit(X_train, y_train)
print(f"Dummy Val Acc: {accuracy_score(y_val, dummy.predict(X_val)):.2f}")
print(f"Dummy Test Acc: {accuracy_score(y_test, dummy.predict(X_test)):.2f}")

# Avaliação de diferentes kernels no SVM
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=seed)
    scores = cross_val_score(svm, X_train, y_train, cv=5)
    print(f"Kernel: {kernel}, CV Mean Acc: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")

# SVM com melhor kernel encontrado (ex: 'linear')
best_kernel = 'linear'
svm_best = SVC(kernel=best_kernel, random_state=seed)
svm_best.fit(X_train, y_train)
print(f"SVM ({best_kernel}) Val Acc: {accuracy_score(y_val, svm_best.predict(X_val)):.2f}")
print(f"SVM ({best_kernel}) Test Acc: {accuracy_score(y_test, svm_best.predict(X_test)):.2f}")

# Avaliação de diferentes Naive Bayes
methods = [GaussianNB, BernoulliNB]
for method in methods:
    model = method()
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{method.__name__} CV Mean Acc: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")

# Ajuste do GaussianNB com GridSearchCV para otimização do var_smoothing
params = {'var_smoothing': np.logspace(-11, -1, 50)}
grid_nb = GridSearchCV(GaussianNB(), param_grid=params, cv=5)
grid_nb.fit(X_train, y_train)

print("Melhor parâmetro para var_smoothing:", grid_nb.best_params_)
print("Melhor média de acurácia (CV):", grid_nb.best_score_)

# Avaliação com dados de validação e teste
y_val_pred = grid_nb.predict(X_val)
y_test_pred = grid_nb.predict(X_test)

print(f"GaussianNB (tuned) Val Acc: {accuracy_score(y_val, y_val_pred):.2f}")
print(f"GaussianNB (tuned) Test Acc: {accuracy_score(y_test, y_test_pred):.2f}")

# SVM final com ajuste fino
svm_final = SVC(kernel='linear', C=0.5, class_weight='balanced', random_state=seed)
svm_final.fit(X_train, y_train)
print(f"SVM Final Val Acc: {accuracy_score(y_val, svm_final.predict(X_val)):.2f}")
print(f"SVM Final Test Acc: {accuracy_score(y_test, svm_final.predict(X_test)):.2f}")
