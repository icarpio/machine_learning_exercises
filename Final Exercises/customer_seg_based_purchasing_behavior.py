"""
Agrupación de clientes según comportamientos de compra


🎯 Objetivo:

Simular un conjunto de datos de clientes con características comerciales, aplicar técnicas de machine learning no supervisado 
para segmentarlos mediante agrupamiento (clustering), y visualizar los resultados para analizar distintos perfiles de cliente. 
Todo el proyecto debe estar organizado usando clases.


📦 Requisitos:

Utiliza las librerías:

numpy
matplotlib
sklearn.cluster.KMeans
sklearn.preprocessing.StandardScaler

Estructura tu código en tres clases:

SimuladorClientes
ModeloSegmentacionClientes
TestSegmentacionClientes

🧩 Clase SimuladorClientes

Esta clase debe simular los datos de clientes con las siguientes características:

Atributos esperados (por cliente):
Monto gastado: valor entre 100 y 10,000.
Frecuencia de compras: entre 1 y 100.
Categorías preferidas: 3 valores aleatorios entre 1 y 5 (representando número de compras por categoría).

Implementa el método:

def generar_datos(self) -> np.ndarray
Este método debe devolver un array de 200 muestras, cada una con 3 columnas:

Monto gastado
Frecuencia de compras
Total de categorías preferidas (suma de los 3 valores generados)

🧠 Clase ModeloSegmentacionClientes

Esta clase debe encargarse de entrenar el modelo y realizar predicciones.

Atributos:

n_clusters: número de grupos a formar (por defecto: 3).
scaler: instancia de StandardScaler.
modelo: instancia de KMeans.

Métodos requeridos:

entrenar(datos: np.ndarray) -> None:

Escala los datos con StandardScaler.
Ajusta el modelo KMeans.
Guarda los datos escalados como atributo para futuras visualizaciones.

predecir(cliente_nuevo: list) -> int:

Recibe un nuevo cliente (3 características).
Escala sus datos.
Devuelve el número de cluster al que pertenece.

🧪 Clase TestSegmentacionClientes

Clase para integrar y probar todo el sistema. Implementa el método:

def ejecutar(self) -> None

Este método debe:

Crear una instancia de SimuladorClientes y generar los datos.
Instanciar ModeloSegmentacionClientes con 3 clusters.
Entrenar el modelo con los datos simulados.
Mostrar los primeros 5 registros de los datos simulados.

Predecir el cluster para un nuevo cliente con los siguientes datos:

cliente_nuevo = [2000, 10, 12]

(Significa: gastó 2000, compra 10 veces, tiene 12 compras sumadas en sus categorías preferidas).

6. Mostrar por pantalla el cluster al que pertenece este nuevo cliente.
  
7. Incluye una visualización de los datos segmentados usando matplotlib.

Representa los clientes en un gráfico de dispersión donde:

El eje X es el monto gastado.
El eje Y es la frecuencia de compras.
Los puntos se colorean según el cluster al que pertenecen (usa modelo.modelo.labels_ para obtenerlos).
Añade etiquetas, título, y barra de color.

💡 Consejos para el alumno

Usa np.column_stack para combinar varias columnas en un array.
Escalar los datos es fundamental en clustering: sin esto, las variables dominantes como “monto gastado” podrían sesgar los grupos.
Usa KMeans(n_clusters=3, random_state=42) para asegurar reproducibilidad.

🧪 Ejemplo de uso

test = TestSegmentacionClientes()
test.ejecutar()


Salida esperada

Primeros 5 registros de datos simulados:
[[3.80794718e+03 2.40000000e+01 1.00000000e+01]
 [9.51207163e+03 7.50000000e+01 1.30000000e+01]
 [7.34674002e+03 7.20000000e+01 9.00000000e+00]
 [6.02671899e+03 3.60000000e+01 1.00000000e+01]
 [1.64458454e+03 3.80000000e+01 9.00000000e+00]]
Modelo entrenado con 3 clusters.
El nuevo cliente pertenece al cluster: 1    
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ===============================================================
# 1. CLASE: SimuladorClientes
# ===============================================================
class SimuladorClientes:
    """
    Esta clase genera datos simulados para clientes.
    Cada cliente tendrá:
    - Monto gastado (100 a 10,000)
    - Frecuencia de compras (1 a 100)
    - Categorías preferidas: suma de 3 valores entre 1 y 5
    """

    def __init__(self, n=200, seed=42):
        self.n = n
        self.seed = seed

    def generar_datos(self) -> np.ndarray:
        """
        Genera un array NumPy de tamaño (n, 3)
        Columnas:
        1) Monto gastado
        2) Frecuencia de compras
        3) Total de categorías preferidas
        """
        np.random.seed(self.seed)

        monto = np.random.uniform(100, 10000, self.n)
        frecuencia = np.random.randint(1, 101, self.n)

        # Tres categorías aleatorias y luego sumamos
        cat1 = np.random.randint(1, 6, self.n)
        cat2 = np.random.randint(1, 6, self.n)
        cat3 = np.random.randint(1, 6, self.n)
        categorias_total = cat1 + cat2 + cat3

        datos = np.column_stack((monto, frecuencia, categorias_total))
        return datos


# ===============================================================
# 2. CLASE: ModeloSegmentacionClientes
# ===============================================================
class ModeloSegmentacionClientes:
    """
    Realiza el clustering usando KMeans.
    Incluye escalado de datos y predicción para nuevos puntos.
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.modelo = KMeans(n_clusters=n_clusters, random_state=42)
        self.datos_escalados = None

    def entrenar(self, datos: np.ndarray) -> None:
        """
        Escala los datos y entrena KMeans.
        """
        self.datos_escalados = self.scaler.fit_transform(datos)
        self.modelo.fit(self.datos_escalados)
        print(f"Modelo entrenado con {self.n_clusters} clusters.")

    def predecir(self, cliente_nuevo: list) -> int:
        """
        Recibe una lista de 3 valores [monto, frecuencia, categorias].
        Devuelve el número de cluster.
        """
        nuevo_escalado = self.scaler.transform([cliente_nuevo])
        cluster = self.modelo.predict(nuevo_escalado)[0]
        return cluster


# ===============================================================
# 3. CLASE: TestSegmentacionClientes
# ===============================================================
class TestSegmentacionClientes:
    """
    Clase que une todo el flujo: simulación, modelo, predicción y visualización.
    """

    def ejecutar(self) -> None:
        # ---- 1. Generar datos ----
        simulador = SimuladorClientes()
        datos = simulador.generar_datos()

        print("Primeros 5 registros de datos simulados:")
        print(datos[:5], "\n")

        # ---- 2. Crear y entrenar el modelo ----
        modelo = ModeloSegmentacionClientes(n_clusters=3)
        modelo.entrenar(datos)

        # ---- 3. Predecir un nuevo cliente ----
        cliente_nuevo = [2000, 10, 12]
        cluster = modelo.predecir(cliente_nuevo)

        print(f"El nuevo cliente pertenece al cluster: {cluster}")

        # ---- 4. Visualización ----
        plt.figure(figsize=(10, 6))

        # Ejes X y Y: monto gastado y frecuencia
        plt.scatter(
            datos[:, 0], datos[:, 1],
            c=modelo.modelo.labels_, cmap='viridis', s=50
        )

        plt.colorbar(label="Cluster asignado")
        plt.xlabel("Monto gastado")
        plt.ylabel("Frecuencia de compras")
        plt.title("Segmentación de Clientes mediante K-Means")
        plt.grid(True)
        plt.show()


# ===============================================================
# EJECUCIÓN DIRECTA
# ===============================================================
if __name__ == "__main__":
    test = TestSegmentacionClientes()
    test.ejecutar()
