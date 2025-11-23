"""
Segmentación de Clientes y Predicción de Compra

Contexto

Eres analista de datos en una empresa de comercio electrónico que quiere mejorar su estrategia de marketing mediante la segmentación de clientes 
y la predicción de su intención de compra.

Tu objetivo es crear un sistema que:

Genere datos sintéticos representativos de clientes reales, con variables como cuánto gastan, cuántas compras hacen, y con qué frecuencia compran.
Segmente a los clientes en grupos similares usando un algoritmo de clustering.
Entrene un modelo predictivo para estimar si un cliente comprará en el próximo mes basándose en sus características y el segmento al que pertenece.
Visualice los segmentos y la probabilidad de compra para facilitar la interpretación de los resultados.

Datos proporcionados y estructura

Clase CustomerDataGenerator

Esta clase debe generar un DataFrame con 300 clientes sintéticos, cada uno con estas columnas:

total_spent: Dinero total gastado por el cliente, en euros (valor aleatorio entre 50 y 1500).
total_purchases: Número total de compras realizadas (entero entre 1 y 50).
purchase_frequency: Frecuencia de compra mensual (valor entre 0.5 y 10).
will_buy_next_month: Etiqueta binaria (1 o 0) que indica si el cliente comprará el próximo mes. 
La regla para asignar 1 es: si total_spent > 500 y purchase_frequency > 4, el cliente comprará (1), si no, no comprará (0).

Modelado

Clase CustomerSegmentationModel

Esta clase debe:

Recibir el DataFrame generado.
Segmentar clientes en 3 grupos usando KMeans con las variables total_spent, total_purchases y purchase_frequency.
Añadir la columna customer_segment al DataFrame con el número de segmento asignado a cada cliente.
Entrenar un modelo de regresión logística para predecir will_buy_next_month, usando como variables las originales más la segmentación (transformada en variables dummy).
Proveer métodos para obtener la precisión del modelo y la matriz de confusión.

Visualizaciones

Función graficar_segmentos(data):

Genera un scatter plot de total_spent vs purchase_frequency.
Usa colores diferentes para cada segmento.
Añade leyenda, etiquetas y título descriptivo.

Función graficar_probabilidad_compra(modelo):

Muestra cómo varía la probabilidad de compra del cliente en función del gasto total (total_spent), manteniendo constantes total_purchases=25 y purchase_frequency=5.
Dibuja la curva de probabilidad predicha por el modelo de regresión logística.

Indicaciones numéricas y técnicas

Número de muestras: 300.
Número de clusters para KMeans: 3.
Random seed: 42 para reproducibilidad.
División de datos para entrenamiento/prueba: 80% / 20%.
Iteraciones máximas para la regresión logística: 500.
Uso solo de numpy, pandas, sklearn y matplotlib

Ejemplo de uso

# 1. Generar datos
generador = CustomerDataGenerator()
datos_clientes = generador.generate(300)
 
# 2. Crear modelo
modelo = CustomerSegmentationModel(datos_clientes)
modelo.segment_customers()
modelo.train_model()
 
# 3. Resultados
print("Precisión del modelo:", modelo.get_accuracy())
print("Matriz de confusión:\n", modelo.get_confusion_matrix())
 
# 4. Visualizaciones
graficar_segmentos(modelo.data)
graficar_probabilidad_compra(modelo.model)


Salida esperada

Precisión del modelo: 0.8833333333333333
Matriz de confusión:
 [[30  2]
 [ 5 23]]

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# 1. GENERADOR DE DATOS
# ============================================================
class CustomerDataGenerator:

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate(self, num_samples=300):
        total_spent = np.random.uniform(50, 1500, num_samples)
        total_purchases = np.random.randint(1, 50, num_samples)
        purchase_frequency = np.random.uniform(0.5, 10, num_samples)

        # Regla para determinar si comprará el próximo mes
        will_buy_next_month = np.where(
            (total_spent > 500) & (purchase_frequency > 4),
            1,
            0
        )

        data = pd.DataFrame({
            "total_spent": total_spent,
            "total_purchases": total_purchases,
            "purchase_frequency": purchase_frequency,
            "will_buy_next_month": will_buy_next_month
        })

        return data

# ============================================================
# 2. MODELO DE SEGMENTACIÓN + REGRESIÓN LOGÍSTICA
# ============================================================
class CustomerSegmentationModel:

    def __init__(self, data):
        self.data = data.copy()
        self.model = None

    def segment_customers(self, n_clusters=3):
        features = self.data[["total_spent", "total_purchases", "purchase_frequency"]]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        segments = kmeans.fit_predict(features)

        self.data["customer_segment"] = segments

    def train_model(self):
        # Crear variables dummy para el segmento
        df = pd.get_dummies(self.data, columns=["customer_segment"], drop_first=True)

        X = df.drop("will_buy_next_month", axis=1)
        y = df["will_buy_next_month"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)

    def get_accuracy(self):
        return self.accuracy

    def get_confusion_matrix(self):
        return self.conf_matrix

# ============================================================
# 3. GRÁFICO DE SEGMENTOS
# ============================================================
def graficar_segmentos(data):
    plt.figure(figsize=(8, 5))

    for seg in data["customer_segment"].unique():
        df_seg = data[data["customer_segment"] == seg]
        plt.scatter(
            df_seg["total_spent"],
            df_seg["purchase_frequency"],
            label=f"Segmento {seg}",
            alpha=0.7
        )

    plt.xlabel("Total Spent (€)")
    plt.ylabel("Purchase Frequency (mes)")
    plt.title("Segmentación de Clientes – KMeans")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# 4. GRÁFICO DE PROBABILIDAD DE COMPRA
# ============================================================
def graficar_probabilidad_compra(model):
    total_spent_range = np.linspace(50, 1500, 200)
    total_purchases = 25
    purchase_frequency = 5

    # Segmento estimado: usar medias ficticias (pues no conocemos asignación real)
    # -> se asignan como 0 las columnas dummy para simplificar
    df_pred = pd.DataFrame({
        "total_spent": total_spent_range,
        "total_purchases": total_purchases,
        "purchase_frequency": purchase_frequency,
        "customer_segment_1": 0,
        "customer_segment_2": 0
    })

    prob = model.predict_proba(df_pred)[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(total_spent_range, prob, color="blue")
    plt.xlabel("Total Spent (€)")
    plt.ylabel("Probabilidad de Compra")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title("Probabilidad de Compra según Total Spent")
    plt.show()

# ============================================================
# EJEMPLO DE USO COMPLETO
# ============================================================
if __name__ == "__main__":

    # 1. Generar datos
    generador = CustomerDataGenerator()
    datos_clientes = generador.generate(300)

    # 2. Crear modelo
    modelo = CustomerSegmentationModel(datos_clientes)
    modelo.segment_customers()
    modelo.train_model()

    # 3. Resultados
    print("Precisión del modelo:", modelo.get_accuracy())
    print("Matriz de confusión:\n", modelo.get_confusion_matrix())

    # 4. Visualizaciones
    graficar_segmentos(modelo.data)
    graficar_probabilidad_compra(modelo.model)
