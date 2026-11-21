import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions

# Se genera un conjunto de datos sencillo
X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)

# Se ddivide en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para entrenar y visualizar resultados
def train_and_plot_svm(kernel, C=1.0, gamma='scale'):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print(f"Resultados con kernel={kernel}, C={C}, gamma={gamma}")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Puntaje F1: {f1_score(y_test, y_pred):.4f}\n")

    plt.figure(figsize=(6,4))
    plot_decision_regions(X, y, clf=svm_model, legend=2)
    plt.title(f"SVM con kernel={kernel}")
    plt.show()

# Probamos diferentes kernels
train_and_plot_svm(kernel='linear')
train_and_plot_svm(kernel='rbf')
train_and_plot_svm(kernel='poly', C=1, gamma='scale')

# Definimos la cuadrícula de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Realizamos la búsqueda en cuadrícula con validación cruzada
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Mostramos los mejores hiperparámetros
print("Mejores hiperparámetros:", grid_search.best_params_)

# Evaluamos el mejor modelo
y_pred_best = grid_search.best_estimator_.predict(X_test)
print("Resultados con los mejores hiperparámetros")
print(f"Precisión: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"Puntaje F1: {f1_score(y_test, y_pred_best):.4f}")

# Visualizamos la mejor clasificación
plt.figure(figsize=(6,4))
plot_decision_regions(X, y, clf=grid_search.best_estimator_, legend=2)
plt.title("Mejor modelo SVM")
plt.show()
