
"""
💡 ¿Qué es K-Medias (K-Means)?

K-Means es un algoritmo de aprendizaje no supervisado (no necesita etiquetas reales para entrenar) que agrupa datos en un número k de grupos (clusters).

🔍 Idea básica:

Le dices al algoritmo cuántos grupos (k) quieres.
K-Means elige al azar k puntos como centroides (los “centros” de cada grupo).
Cada punto del dataset se asigna al centroide más cercano.
Luego, el algoritmo recalcula los centroides (promediando los puntos asignados).
Repite los pasos 3 y 4 hasta que los centroides no cambien mucho (converge).

👉 El resultado: cada muestra del dataset pertenece a uno de los k grupos.


🧮 Métricas que mediremos:
Métrica	                                  Qué mide	                        Interpretación

Inertia	             Suma de distancias de los puntos a su centroide	    Cuanto más pequeña, mejor
Silhouette Score	 Qué tan separados están los clusters	                Entre -1 y 1 (más alto = mejor)
Adjusted Rand Score	 Compara con etiquetas reales	                        Entre 0 y 1 (1 = perfecto)

"""



"""
Ejercicio Algoritmo de k-medias
El objetivo es implementar una función que:

Entrene un modelo K-Means (KMeans de sklearn.cluster).

Agrupe los datos en k clusters.

Evalúe el rendimiento del modelo utilizando:

Inercia (inertia_): mide qué tan bien se agrupan los datos.
Puntuación de Silueta (silhouette_score): evalúa qué tan bien están separados los clusters.
Precisión ajustada (adjusted_rand_score): mide la similitud con las etiquetas reales.
Devuelva los resultados en un diccionario.
Supervise la implementación con pruebas unitarias (unittest).

Instrucciones

Implementa una función llamada entrenar_y_evaluar_kmeans(X, y, k) que:

Entrene un modelo KMeans(n_clusters=k, random_state=42).
Asigne cada muestra a un cluster (labels_).
Calcule las métricas de evaluación mencionadas.

Devuelva un diccionario con:

"clusters": Array con las asignaciones de cluster para cada muestra.
"inertia": Suma de las distancias al centroide más cercano.
inertia = modelo.inertia_
"silhouette_score": Puntuación de silueta (qué tan bien se separan los clusters).
silhouette = silhouette_score(X, clusters)
"adjusted_rand_score": Similaridad con las etiquetas reales.
rand_score = adjusted_rand_score(y, clusters)

Usa el dataset de flores iris de sklearn.datasets.
Asegúrate de que k=3 (correspondiente a las 3 clases reales de Iris).
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import load_iris



def entrenar_y_evaluar_kmeans(X, y, k):
    # 1. Entrenar el modelo K-Means
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X)
    
    # 2. Obtener los clusters asignados
    clusters = modelo.labels_
    
    # 3. Calcular las métricas
    inertia = modelo.inertia_
    silhouette = silhouette_score(X, clusters)
    rand_score = adjusted_rand_score(y, clusters)
    
    # 4. Devolver resultados en un diccionario
    resultados = {
        "clusters": clusters,
        "inertia": inertia,
        "silhouette_score": silhouette,
        "adjusted_rand_score": rand_score
    }
    
    return resultados


# 🧪 Prueba del código con el dataset Iris
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Cargar el dataset de Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Entrenar y evaluar
    resultados = entrenar_y_evaluar_kmeans(X, y, k=3)

    # Mostrar resultados
    print("Inercia del modelo:", resultados["inertia"])
    print("Silhouette Score:", resultados["silhouette_score"])
    print("Adjusted Rand Score:", resultados["adjusted_rand_score"])
    print("Clusters asignados:\n", resultados["clusters"][:10])