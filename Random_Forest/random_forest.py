"""
🌲 ¿Qué es Random Forest en Machine Learning?

Random Forest es un modelo de aprendizaje supervisado basado en el ensamble de árboles de decisión.
La idea principal es combinar muchos árboles de decisión (cada uno entrenado sobre una muestra aleatoria de los datos y de las características) y luego promediar sus predicciones (en clasificación, votación mayoritaria).

Ventajas:

Reduce el sobreajuste (overfitting) comparado con un árbol único.

Maneja bien datos numéricos y categóricos.

Proporciona estimaciones de importancia de características.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test):
    # 1. Crear el modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 2. Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # 3. Hacer predicciones
    predicciones = modelo.predict(X_test)
    
    # 4. Calcular métricas
    accuracy = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)
    
    # 5. Devolver resultados
    resultados = {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz,
        "reporte": reporte
    }
    
    return resultados



# Prueba del modelo
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from solution import entrenar_y_evaluar_random_forest

# Cargar dataset
wine = load_wine()
X = wine.data
y = wine.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ejecutar función
resultados = entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test)

# Mostrar resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])
