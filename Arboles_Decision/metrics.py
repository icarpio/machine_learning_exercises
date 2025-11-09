"""
*** Métricas en árboles de decisión ***

El objetivo de este ejercicio es que los estudiantes implementen una función que:
Entrene un árbol de decisión usando DecisionTreeClassifier de sklearn.
Haga predicciones en un conjunto de prueba.
Evalúe el modelo utilizando métricas como precisión (accuracy), matriz de confusión y reporte de clasificación.
Pase pruebas unitarias (unittest) que validen el funcionamiento correcto del código.


Instrucciones

Implementa una función llamada entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test) que:

- Entrene un modelo DecisionTreeClassifier con los datos de entrenamiento (X_train, y_train).
- Prediga los valores de X_test.

Evalúe el modelo usando:

-Precisión (accuracy_score)
-Matriz de confusión (confusion_matrix)
-Reporte de clasificación (classification_report)

evuelva un diccionario con:

-predicciones: Un array con las predicciones del modelo.
-accuracy: Un número flotante con la precisión.
-matriz_confusion: Una matriz de confusión.
-reporte: Un string con el reporte de clasificación.

Usa random_state=42 en DecisionTreeClassifier para reproducibilidad.
Prueba la función con el dataset Iris, asegurando que el modelo tenga al menos 85% de precisión en los datos de prueba.

"""


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test):
    """
    Entrena un árbol de decisión y evalúa su desempeño en un conjunto de prueba.
    
    Parámetros:
    - X_train: array de NumPy con las características de entrenamiento
    - y_train: array de NumPy con las etiquetas de entrenamiento
    - X_test: array de NumPy con las características de prueba
    - y_test: array de NumPy con las etiquetas verdaderas de prueba
    
    Retorna:
    - Diccionario con:
        'predicciones': array de predicciones del modelo
        'accuracy': precisión del modelo
        'matriz_confusion': matriz de confusión
        'reporte': reporte de clasificación con nombres de clases
    """
    
    # Nombres de las clases para el dataset Iris
    nombres_clases = ['Setosa', 'Versicolor', 'Virginica']
    
    # 1️⃣ Crear el modelo de árbol de decisión
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 2️⃣ Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # 3️⃣ Hacer predicciones sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)
    
    # 4️⃣ Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    
    # 5️⃣ Generar reporte de clasificación con nombres de clases
    reporte = classification_report(
        y_test,
        predicciones,
        labels=[0, 1, 2],           # índices de las clases
        target_names=nombres_clases # nombres legibles en el reporte
    )
    
    # 6️⃣ Devolver resultados en un diccionario
    return {
        'predicciones': np.array(predicciones),
        'accuracy': accuracy,
        'matriz_confusion': matriz_confusion,
        'reporte': reporte
    }

#🔹 Ejemplo de uso

# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Llamar a la función
resultados = entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test)

# Mostrar resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])