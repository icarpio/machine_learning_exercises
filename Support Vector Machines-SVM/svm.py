"""
Las Máquinas de Vectores de Soporte (en inglés, Support Vector Machines — SVM) son uno de los algoritmos más potentes y populares del aprendizaje supervisado, 
especialmente para clasificación (aunque también se pueden usar para regresión y detección de outliers).

🧠 Idea principal

Imagina que tienes dos grupos de puntos (clases) en un plano:

🔵 Clase A
🔴 Clase B

El objetivo del SVM es encontrar una línea (o un hiperplano en dimensiones mayores) que separe ambas clases lo mejor posible.

Pero no cualquier línea:
👉 Se busca la que deja el mayor margen posible entre ambas clases.

Ese margen está definido por los puntos más cercanos al límite, llamados vectores de soporte — de ahí el nombre del modelo.

📏 Concepto clave: el margen máximo

El SVM busca un hiperplano óptimo que:

Separe las clases correctamente (si es posible).
Maximice la distancia entre las clases (margen).
Matemáticamente, el SVM resuelve un problema de optimización convexa para maximizar ese margen bajo ciertas restricciones.

🔄 Cuando los datos no son lineales

En muchos casos, las clases no se pueden separar con una línea recta.
Por ejemplo:

🔵🔵🔵🔵🔵
🔵🔵🔵🔵🔵
    🔴🔴🔴
    🔴🔴🔴


Ahí entra en juego el truco del kernel (kernel trick).

✨ El truco del kernel

Consiste en transformar los datos a un espacio de mayor dimensión, donde sí se puedan separar linealmente, sin necesidad de calcular esa transformación explícitamente.

Ejemplo:

En 2D no hay una línea que separe bien los puntos.
En 3D (tras una transformación con un kernel), puede existir un plano separador perfecto.

Los kernels más usados:

linear: separa con una línea recta.
poly: usa funciones polinomiales.
rbf o gaussian: transforma los datos con una función gaussiana (muy potente).
sigmoid: similar a una red neuronal.

"""



"""
Ejecicio SVM - Máquinas de vectores de soporte

Objetivo

El objetivo es implementar una función que:

Entrene un modelo de Máquina de Soporte Vectorial (SVM) usando SVC de sklearn.svm.
Realice predicciones en un conjunto de prueba.

Evalúe el modelo con las siguientes métricas:

Precisión (accuracy_score).
Matriz de confusión (confusion_matrix).
Reporte de clasificación (classification_report).
Devuelva los resultados en un diccionario.
Supervise la implementación con pruebas unitarias (unittest).



Instrucciones

Implementa una función llamada entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test) que:
Entrene un modelo SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42).
Prediga los valores de X_test.
Calcule las métricas de evaluación mencionadas.

Devuelva un diccionario con:

"predicciones": Array de predicciones del modelo.
"accuracy": Precisión del modelo en los datos de prueba.
"matriz_confusion": Matriz de confusión.
"reporte": Reporte de clasificación.

Usa el dataset de digits de sklearn.datasets, que contiene imágenes de números escritos a mano.
Asegúrate de que el modelo tenga al menos 90% de precisión en los datos de prueba.

"""

"""
SVM (Support Vector Machine) es un modelo de clasificación supervisada que busca encontrar la frontera óptima (hiperplano) que separa las clases 
maximizando el margen entre los puntos más cercanos (vectores de soporte).
Usaremos un kernel RBF (radial basis function), que permite separar datos no lineales transformando el espacio de características.
"""


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo SVM con kernel RBF sobre los datos dados.

    Parámetros:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba

    Retorna:
        Un diccionario con:
            - "predicciones": array de predicciones
            - "accuracy": precisión del modelo
            - "matriz_confusion": matriz de confusión
            - "reporte": reporte de clasificación
    """
    # 1️⃣ Crear el modelo SVM
    modelo = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)

    # 2️⃣ Entrenar el modelo
    modelo.fit(X_train, y_train)

    # 3️⃣ Hacer predicciones
    predicciones = modelo.predict(X_test)

    # 4️⃣ Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)

    # 5️⃣ Devolver resultados
    resultados = {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz,
        "reporte": reporte
    }

    return resultados

#Ejemplo de uso

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from solution import entrenar_y_evaluar_svm

# Cargar dataset
digits = load_digits()
X = digits.data
y = digits.target

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar y evaluar
resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)

# Mostrar resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])