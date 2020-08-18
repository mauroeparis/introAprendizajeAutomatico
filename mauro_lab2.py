# %%markdown
## Laboratorio 2: Armado de un esquema de aprendizaje automático

# %%
import numpy as np
import pandas as pd

# TODO: Agregar las librerías que hagan falta
from sklearn.model_selection import train_test_split

# %%markdown
### Carga de datos y división en entrenamiento y evaluación

La celda siguiente se encarga de la carga de datos (haciendo uso de pandas).
Estos serán los que se trabajarán en el resto del laboratorio.

# %%
dataset = pd.read_csv("./data/loan_data.csv", comment="#")

# División entre instancias y etiquetas
X, y = dataset.iloc[:, 1:], dataset.TARGET

# división entre entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

RANDOM_STATE_SEED = 42

# %%markdown
### Ejercicio 1: Descripción de los Datos y la Tarea

Responder las siguientes preguntas:

1. ¿De qué se trata el conjunto de datos?
2. ¿Cuál es la variable objetivo que hay que predecir? ¿Qué significado tiene?
3. ¿Qué información (atributos) hay disponible para hacer la predicción?
4. ¿Qué atributos imagina ud. que son los más determinantes para la predicción?

# %%markdown
#### Respuestas:

**1.** El dataset contiene información del desarrollo de 5960 prestamos
inmobiliarios.

**2.** La variable objetivo es `BAD` y es una variable binaria que determina si
el aplicante al prestamo no pudo seguir pagandolo o tuvo serios retrasos al
momento de hacerlo.

**3.** Tenemos los siguientes atributos para hacer la predicción:

- _TARGET (BAD)_: Si el cliente no pudo pagar el prestamo. (1 o 0)
- _LOAN_: Tamaño del prestamo
- _MORTDUE_: Cuanto debe en la hipoteca actual
- _VALUE_: Valor de la propiedad actual
- _YOJ_: Años que lleva en el trabajo actual
- _DEROG_: Número de informes despectivos (negativos?)
- _DELINQ_: Número de líneas de crédito morosas
- _CLAGE_: Cantidad de meses desde la última transacción
- _NINQ_: Número de lineas de crédito recientes
- _CLNO_: Número de lineas de crédito
- _DEBTINC_: El ratio de deuda a salario

**4.** Imagino que `DELINQ`, `DEBTINC` y `LOAN` deben ser los que más afecten al
resultado.

# %%markdown
### Ejercicio 2: Predicción con Modelos Lineales

En este ejercicio se entrenarán modelos lineales de clasificación para predecir
 la variable objetivo.

Para ello, deberán utilizar la clase SGDClassifier de scikit-learn.

Documentación:
- https://scikit-learn.org/stable/modules/sgd.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html


# %%markdown
### Ejercicio 2.1: SGDClassifier con hiperparámetros por defecto

Entrenar y evaluar el clasificador SGDClassifier usando los valores por omisión
 de scikit-learn para todos los parámetros. Únicamente **fijar la semilla
  aleatoria** para hacer repetible el experimento.

Evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de
 **evaluación**, reportando:
- Accuracy
- Precision
- Recall
- F1
- matriz de confusión

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(random_state=RANDOM_STATE_SEED)
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# %%
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)


print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
print('Precision Score : ' + str(precision_score(y_test, y_pred)))
print('Recall Score : ' + str(recall_score(y_test, y_pred)))
print('F1 Score : ' + str(f1_score(y_test, y_pred)))

# %%
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print('True Negative: ' + str(tn))
print('False Positive: ' + str(fp))
print('False Negative: ' + str(fn))
print('True Positive: ' + str(tp))


# %%markdown
### Ejercicio 2.2: Ajuste de Hiperparámetros

Seleccionar valores para los hiperparámetros principales del SGDClassifier.
Como mínimo, probar diferentes funciones de loss, tasas de entrenamiento
y tasas de regularización.

Para ello, usar grid-search y 5-fold cross-validation sobre el conjunto de
entrenamiento para explorar muchas combinaciones posibles de valores.

Reportar accuracy promedio y varianza para todas las configuraciones.

Para la mejor configuración encontrada, evaluar sobre el conjunto de
**entrenamiento** y sobre el conjunto de **evaluación**, reportando:

- Accuracy
- Precision
- Recall
- F1
- matriz de confusión

Documentación:
- https://scikit-learn.org/stable/modules/grid_search.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# %%
from sklearn.model_selection import GridSearchCV

grid_values = {
    'sgdclassifier__loss': ['perceptron', 'hinge', 'log'],
    'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
    'sgdclassifier__alpha': [0.0001, 0.001, 0.0005],
    'sgdclassifier__learning_rate': [
        'optimal', 'constant', 'invscaling', 'adaptive'],
    'sgdclassifier__eta0': [1, 1.5, 2]
}
grid_clf_acc = GridSearchCV(
    clf,
    param_grid=grid_values,
    scoring='recall'
)
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))

# %%markdown
## Ejercicio 3: Árboles de Decisión

En este ejercicio se entrenarán árboles de decisión para predecir la variable
objetivo.

Para ello, deberán utilizar la clase DecisionTreeClassifier de scikit-learn.

Documentación:
- https://scikit-learn.org/stable/modules/tree.html
  - https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

# %%markdown
### Ejercicio 3.1: DecisionTreeClassifier con hiperparámetros por defecto

Entrenar y evaluar el clasificador DecisionTreeClassifier usando los valores
 por omisión de scikit-learn para todos los parámetros. Únicamente
  **fijar la semilla aleatoria** para hacer repetible el experimento.

Evaluar sobre el conjunto de **entrenamiento** y sobre el conjunto de
**evaluación**, reportando:

- Accuracy
- Precision
- Recall
- F1
- matriz de confusión

# %%
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

y_test_pred = clf_tree.predict(X_test)

# %%
print('Accuracy Score : ' + str(accuracy_score(y_test, y_test_pred)))
print('Precision Score : ' + str(precision_score(y_test, y_test_pred)))
print('Recall Score : ' + str(recall_score(y_test, y_test_pred)))
print('F1 Score : ' + str(f1_score(y_test, y_test_pred)))

# %%
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

print('True Negative: ' + str(tn))
print('False Positive: ' + str(fp))
print('False Negative: ' + str(fn))
print('True Positive: ' + str(tp))

# %%markdown
### Ejercicio 3.2: Ajuste de Hiperparámetros

Seleccionar valores para los hiperparámetros principales del
DecisionTreeClassifier. Como mínimo, probar diferentes criterios de
partición (criterion), profundidad máxima del árbol (max_depth), y
cantidad mínima de samples por hoja (min_samples_leaf).

Para ello, usar grid-search y 5-fold cross-validation sobre el
conjunto de entrenamiento para explorar muchas combinaciones
posibles de valores.

Reportar accuracy promedio y varianza para todas las configuraciones.

Para la mejor configuración encontrada, evaluar sobre el conjunto
de **entrenamiento** y sobre el conjunto de **evaluación**, reportando:

- Accuracy
- Precision
- Recall
- F1
- matriz de confusión


Documentación:
- https://scikit-learn.org/stable/modules/grid_search.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# %%
grid_values = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 5, 15, 30],
    'min_samples_leaf': [1, 2, 3, 5],
}
grid_clf_acc = GridSearchCV(
    clf_tree,
    param_grid=grid_values,
    scoring='recall'
)
grid_clf_acc.fit(X_train, y_train)

#Predict values based on new parameters
y_pred_acc = grid_clf_acc.predict(X_test)

# New Model Evaluation metrics
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_acc)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_acc)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_acc)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_acc)))
