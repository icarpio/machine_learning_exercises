"""
Clasificador inteligente de materiales reciclables
Contexto:
Imagina que trabajas para una empresa de reciclaje inteligente.

Tu tarea consiste en diseñar un sistema que pueda predecir automáticamente si un objeto es papel, plástico o metal, a partir de sus propiedades físicas, usando el algoritmo de k vecinos más cercanos (KNN).

Vas a utilizar Python con las librerías numpy, pandas, matplotlib y sklearn para entrenar y visualizar el modelo.



📦 Objetivo

Implementa las siguientes clases:

1. RecyclableItem

Representa un objeto reciclable con tres atributos:

weight: peso del objeto en gramos
volume: volumen en cm³.
material_type: tipo de material codificado como:

0 para papel
1 para plástico
2 para metal

Método necesario:

to_vector(self): devuelve [weight, volume], útil para alimentar el modelo.

2. RecyclingDataGenerator

Genera objetos sintéticos para entrenar el modelo.

Métodos:

__init__(self, num_samples=150): constructor de la clase:

num_samples: número total de objetos a generar (repartidos entre los tres tipos de material).
generate(self): genera y devuelve una lista de objetos RecyclableItem con las siguientes características:
Papel (0):
Peso: media ≈ 30 g → np.random.normal(30, 5)
Volumen: media ≈ 250 cm³ → np.random.normal(250, 30)
Plástico (1):
Peso: media ≈ 80 g → np.random.normal(80, 10)
Volumen: media ≈ 150 cm³ → np.random.normal(150, 20)
Metal (2):
Peso: media ≈ 150 g → np.random.normal(150, 20)
Volumen: media ≈ 80 cm³ → np.random.normal(80, 10)

3. RecyclableMaterialClassifier

Clasificador que entrena un modelo de KNN.

Métodos:

Constructor de la clase  __init__(self, k=5):
k: número de vecinos más cercanos a usar (por defecto: 5)
fit(records): entrena el modelo con una lista de objetos RecyclableItem.
predict(weight, volume): devuelve el tipo de material predicho (0, 1 o 2) para un nuevo objeto.
evaluate(records): imprime métricas de clasificación (classification_report, confusion_matrix) con un conjunto de prueba.



4. RecyclablePredictionExample

Clase que coordina todo el flujo:

Genera los datos.
Separa en entrenamiento y prueba.
Entrena el clasificador.
Evalúa el rendimiento.
Hace una predicción para un nuevo objeto (por ejemplo, peso = 60, volumen = 180).
Visualiza los datos y las predicciones en un gráfico 2D con colores distintos para cada tipo de material.

"""


# Importamos las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Clase RecyclableItem
# =========================
class RecyclableItem:
    def __init__(self, weight, volume, material_type):
        """
        Representa un objeto reciclable.
        weight: peso en gramos
        volume: volumen en cm³
        material_type: 0=papel, 1=plástico, 2=metal
        """
        self.weight = weight
        self.volume = volume
        self.material_type = material_type

    def to_vector(self):
        """Devuelve [peso, volumen] para usar en el modelo."""
        return [self.weight, self.volume]

# =========================
# 2. Clase RecyclingDataGenerator
# =========================
class RecyclingDataGenerator:
    def __init__(self, num_samples=150):
        """
        Generador de datos sintéticos.
        num_samples: total de objetos a generar
        """
        self.num_samples = num_samples

    def generate(self):
        """
        Genera una lista de objetos RecyclableItem con distribuciones normales según el tipo de material.
        """
        items = []
        samples_per_type = self.num_samples // 3

        # Papel (0)
        for _ in range(samples_per_type):
            weight = np.random.normal(30, 5)
            volume = np.random.normal(250, 30)
            items.append(RecyclableItem(weight, volume, 0))

        # Plástico (1)
        for _ in range(samples_per_type):
            weight = np.random.normal(80, 10)
            volume = np.random.normal(150, 20)
            items.append(RecyclableItem(weight, volume, 1))

        # Metal (2)
        for _ in range(samples_per_type):
            weight = np.random.normal(150, 20)
            volume = np.random.normal(80, 10)
            items.append(RecyclableItem(weight, volume, 2))

        return items

# =========================
# 3. Clase RecyclableMaterialClassifier
# =========================
class RecyclableMaterialClassifier:
    def __init__(self, k=5):
        """
        Clasificador KNN.
        k: número de vecinos
        """
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, records):
        """
        Entrena el modelo con una lista de objetos RecyclableItem.
        """
        X = [r.to_vector() for r in records]
        y = [r.material_type for r in records]
        self.model.fit(X, y)

    def predict(self, weight, volume):
        """
        Predice el tipo de material de un nuevo objeto.
        """
        return self.model.predict([[weight, volume]])[0]

    def evaluate(self, records):
        """
        Evalúa el modelo con un conjunto de prueba.
        Imprime matriz de confusión y reporte de clasificación.
        """
        X_test = [r.to_vector() for r in records]
        y_test = [r.material_type for r in records]
        y_pred = self.model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# =========================
# 4. Clase RecyclablePredictionExample
# =========================
class RecyclablePredictionExample:
    def __init__(self):
        self.generator = RecyclingDataGenerator(num_samples=150)
        self.classifier = RecyclableMaterialClassifier(k=5)

    def run(self):
        # Generamos los datos
        data = self.generator.generate()

        # Separamos en entrenamiento y prueba (70%-30%)
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        # Entrenamos el clasificador
        self.classifier.fit(train_data)

        # Evaluamos en el conjunto de prueba
        print("=== Evaluación del modelo ===")
        self.classifier.evaluate(test_data)

        # Predicción para un nuevo objeto
        new_weight = 60
        new_volume = 180
        predicted_type = self.classifier.predict(new_weight, new_volume)
        material_names = {0: "Papel", 1: "Plástico", 2: "Metal"}
        print("\n📦 Predicción para un nuevo objeto:")
        print(f"   Peso: {new_weight}g, Volumen: {new_volume}cm³")
        print(f"   Tipo estimado: {material_names[predicted_type]}")

        # Visualización 2D
        self.plot_data(train_data, test_data, new_weight, new_volume, predicted_type)

    def plot_data(self, train_data, test_data, new_weight, new_volume, predicted_type):
        """
        Grafica los objetos reciclables y la predicción de un nuevo objeto.
        """
        colors = {0: 'blue', 1: 'green', 2: 'red'}
        labels = {0: 'Papel', 1: 'Plástico', 2: 'Metal'}

        plt.figure(figsize=(8,6))

        # Graficamos los datos de entrenamiento
        for r in train_data:
            plt.scatter(r.weight, r.volume, color=colors[r.material_type], alpha=0.5, label=f"Entrenamiento {labels[r.material_type]}")
        # Graficamos los datos de prueba
        for r in test_data:
            plt.scatter(r.weight, r.volume, edgecolor='k', facecolor='none', s=100, label=f"Prueba {labels[r.material_type]}")

        # Graficamos el nuevo objeto
        plt.scatter(new_weight, new_volume, color=colors[predicted_type], marker='X', s=200, label='Nuevo objeto')

        # Evitamos etiquetas repetidas en la leyenda
        handles, legend_labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlabel('Peso (g)')
        plt.ylabel('Volumen (cm³)')
        plt.title('Clasificación de Materiales Reciclables')
        plt.grid(True)
        plt.show()

# =========================
# Ejemplo de uso
# =========================
if __name__ == "__main__":
    example = RecyclablePredictionExample()
    example.run()
