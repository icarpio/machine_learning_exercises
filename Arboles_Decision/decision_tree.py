import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Importamos el clasificador de árboles de decisión

def entrenar_arbol_decision(X_train, y_train, X_test):
    """
    Esta función entrena un árbol de decisión con los datos de entrenamiento
    y predice las clases para los datos de prueba.
    
    Parámetros:
    - X_train: array de NumPy con las características de entrenamiento
    - y_train: array de NumPy con las etiquetas de entrenamiento
    - X_test: array de NumPy con las características de prueba
    
    Retorna:
    - Array de NumPy con las predicciones para X_test
    """
    
    # 1️⃣ Crear el modelo de árbol de decisión
    # random_state=42 garantiza que los resultados sean reproducibles
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 2️⃣ Entrenar el modelo usando los datos de entrenamiento
    # El modelo aprende patrones de X_train para predecir y_train
    modelo.fit(X_train, y_train)
    
    # 3️⃣ Hacer predicciones sobre los datos de prueba
    # El modelo usa lo que aprendió para predecir las etiquetas de X_test
    predicciones = modelo.predict(X_test)
    
    # 4️⃣ Devolver las predicciones como un array de NumPy
    return np.array(predicciones)

# Datos de ejemplo
X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
y_train = np.array([0, 1, 0, 1])
X_test = np.array([[0, 1], [1, 0]])

# Entrenar y predecir
predicciones = entrenar_arbol_decision(X_train, y_train, X_test)
print(predicciones)  # Salida: array([0, 1])

"""
🔹 Cómo funciona un árbol de decisión

Entrenamiento (fit):
El árbol analiza tus datos de entrada (X_train) y aprende reglas para dividirlos según las características que mejor separan las clases (y_train). Por ejemplo, si tienes datos de flores, el árbol podría aprender reglas como:

“Si el largo del pétalo < 2.5 → Clase A”

“Si el largo del pétalo ≥ 2.5 y el ancho del pétalo < 1 → Clase B”

Predicción (predict):
Una vez entrenado, el árbol puede usar esas reglas para clasificar nuevos datos (X_test).

random_state:
Esto asegura que si entrenas el mismo modelo varias veces con los mismos datos, obtendrás el mismo árbol y predicciones.

"""