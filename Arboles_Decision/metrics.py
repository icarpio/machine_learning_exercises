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


Precision (Precisión)

Qué mide: De todas las veces que el modelo dijo “esta muestra es de la clase X”, cuántas veces acertó.

Ejemplo:

Supongamos que el modelo predijo 10 flores como Setosa, pero solo 8 eran realmente Setosa.
Precision = 8 / 10 = 0.8 → 80% de las predicciones para Setosa fueron correctas.

Recall (Sensibilidad o Exhaustividad)

Qué mide: De todas las muestras que realmente son de la clase X, cuántas el modelo detectó correctamente.

Ejemplo:

Hay 12 flores que son realmente Setosa. El modelo predijo correctamente 8 de ellas.
Recall = 8 / 12 ≈ 0.67 → Detectó el 67% de las Setosa reales.

F1-score

Qué mide: Es un promedio que combina precision y recall, para dar una sola métrica balanceada.

Ejemplo:

Con el ejemplo anterior, precision = 0.8 y recall = 0.67
F1 ≈ 2 * (0.8*0.67)/(0.8+0.67) ≈ 0.73 → Una sola medida que resume el desempeño.

4️⃣ Support (Soporte)

Qué mide: Cuántas muestras reales hay de cada clase.
Ejemplo: Si hay 12 flores Setosa, 10 Versicolor y 8 Virginica, el support nos dice eso para cada clase.

💡 Resumiendo:

-Precision: ¿De todas mis predicciones, cuántas fueron correctas?
-Recall: ¿De todas las muestras reales, cuántas detecté correctamente?
-F1-score: Balance entre precision y recall.
-Support: Cuántas muestras de esa clase había.

"""


import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Para crear árboles de decisión
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Para evaluar el modelo
from sklearn.datasets import load_iris  # Dataset de ejemplo
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba

def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test):
    """
    Función que entrena un árbol de decisión y evalúa su desempeño.

    Parámetros:
    - X_train: características de entrenamiento (inputs)
    - y_train: etiquetas de entrenamiento (lo que queremos predecir)
    - X_test: características de prueba (inputs nuevos)
    - y_test: etiquetas reales de prueba (para comparar con las predicciones)
    
    Retorna un diccionario con:
    - predicciones: lo que el modelo predijo
    - accuracy: precisión del modelo (qué tan bien predijo)
    - matriz_confusion: muestra errores y aciertos por clase
    - reporte: métricas más detalladas por clase
    """

    # 👇 Nombres de las clases del dataset Iris
    nombres_clases = ['Setosa', 'Versicolor', 'Virginica']

    # 1️⃣ Crear el modelo de árbol de decisión
    # random_state=42 asegura que los resultados sean reproducibles
    modelo = DecisionTreeClassifier(random_state=42)

    # 2️⃣ Entrenar el modelo usando los datos de entrenamiento
    # El modelo "aprende" la relación entre X_train y y_train
    modelo.fit(X_train, y_train)

    # 3️⃣ Hacer predicciones sobre los datos de prueba
    predicciones = modelo.predict(X_test)  # Devuelve un array con las clases predichas

    # 4️⃣ Calcular métricas para evaluar el desempeño del modelo
    accuracy = accuracy_score(y_test, predicciones)  # Qué porcentaje de predicciones fueron correctas
    matriz_confusion = confusion_matrix(y_test, predicciones)  # Muestra aciertos y errores por clase

    # 5️⃣ Crear un reporte más detallado
    # Muestra precision, recall y f1-score por cada clase
    # labels=[0,1,2] indica los índices de las clases en y_test
    # target_names=nombres_clases reemplaza los números por nombres legibles
    reporte = classification_report(
        y_test,
        predicciones,
        labels=[0, 1, 2],
        target_names=nombres_clases
    )

    # 6️⃣ Devolver todo en un diccionario para poder usarlo fácilmente
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