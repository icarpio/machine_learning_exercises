"""
Clasificador de jugadores de baloncesto

Objetivo del ejercicio:

Tu misión es construir un modelo inteligente que clasifique a jugadores de baloncesto según su rendimiento en tres categorías: "Bajo", "Medio" y "Alto", 
utilizando para ello sus características físicas y estadísticas de juego.
Usarás el algoritmo de árboles de decisión junto con NumPy, pandas, matplotlib y scikit-learn.


🎯 Contexto del problema

Un equipo de baloncesto ficticio está evaluando a nuevos jugadores y necesita una herramienta que, a partir de la altura, el peso y el promedio de puntos por partido, 
determine automáticamente el nivel de rendimiento del jugador.
Esta herramienta será clave para seleccionar a los mejores candidatos.


🧱 Estructura sugerida de la solución

1. BasketballPlayer

Una clase que representa a cada jugador. Sus atributos son:

height (int): altura en centímetros.
weight (int): peso en kilogramos.
avg_points (float): promedio de puntos por partido.
performance (str): nivel de rendimiento, con valores "Bajo", "Medio" o "Alto".

Método útil:

to_vector(): devuelve [height, weight, avg_points] para ser usado por el modelo.

2. BasketballDataGenerator

Una clase que genera datos sintéticos simulando jugadores reales.

Constructor

__init__(self, num_samples=200)
num_samples: número total de jugadores a generar (por defecto: 200).

Método clave:

generate(): devuelve una lista de objetos BasketballPlayer.

Cada jugador se genera con las siguientes características:

Altura (height): generada con una distribución normal de media 190 cm y desviación estándar 10 cm:
heights = np.random.normal(190, 10, self.num_samples)
Peso (weight): generada con una distribución normal de media 85 kg y desviación estándar 10 kg:

Promedio de puntos por partido (avg_points): generada con una distribución normal de media 10 puntos y desviación estándar 5:

El rendimiento del jugador se asigna de acuerdo al valor del promedio de puntos:

Menos de 8 puntos → "Bajo"
Entre 8 y 15 puntos → "Medio"
Más de 15 puntos → "Alto"


3. BasketballPerformanceClassifier

Encapsula el modelo de árbol de decisión. Métodos clave:

fit(players): entrena el modelo con una lista de jugadores.
predict(height, weight, avg_points): predice el rendimiento de un nuevo jugador.
evaluate(players): imprime la matriz de confusión y el informe de clasificación sobre un conjunto de prueba.



4. BasketballPredictionExample

Contiene el flujo principal:

-Generar datos.
-Dividirlos en entrenamiento y prueba.
-Entrenar y evaluar el clasificador.
-Hacer una predicción para un nuevo jugador (por ejemplo: altura = 198 cm, peso = 92 kg, puntos = 17).
-Visualizar los jugadores usando matplotlib, diferenciando el rendimiento por colores.



📊 Visualización esperada

Un gráfico de dispersión donde cada punto representa un jugador.

El eje X muestra la altura y el eje Y el promedio de puntos.

El color indica el rendimiento:

🔴 Bajo
🟠 Medio
🟢 Alto

Además:

Se debe marcar la posición del jugador nuevo (por ejemplo, con una estrella ⭐ o un punto grande en azul).

✅ Requisitos técnicos

Usa NumPy para generar datos aleatorios.
Usa pandas para crear el DataFrame de visualización.
Usa DecisionTreeClassifier de sklearn.tree.
Representa visualmente los datos con matplotlib.
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# ====================================================
# 1️⃣ Clase que representa a un jugador de baloncesto
# ====================================================
class BasketballPlayer:
    def __init__(self, height, weight, avg_points, performance):
        self.height = height          # Altura en cm
        self.weight = weight          # Peso en kg
        self.avg_points = avg_points  # Promedio de puntos por partido
        self.performance = performance  # "Bajo", "Medio" o "Alto"

    def to_vector(self):
        """Devuelve las características numéricas del jugador"""
        return [self.height, self.weight, self.avg_points]


# ====================================================
# 2️⃣ Generador de datos sintéticos
# ====================================================
class BasketballDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        """Genera jugadores ficticios con valores realistas"""
        heights = np.random.normal(190, 10, self.num_samples)   # Alturas
        weights = np.random.normal(85, 10, self.num_samples)    # Pesos
        avg_points = np.random.normal(10, 5, self.num_samples)  # Puntos

        players = []

        for h, w, p in zip(heights, weights, avg_points):
            # Clasificamos el rendimiento según el promedio de puntos
            if p < 8:
                perf = "Bajo"
            elif 8 <= p <= 15:
                perf = "Medio"
            else:
                perf = "Alto"

            players.append(BasketballPlayer(h, w, p, perf))

        return players


# ====================================================
# 3️⃣ Clasificador basado en Árbol de Decisión
# ====================================================
class BasketballPerformanceClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, players):
        """Entrena el modelo con los datos de los jugadores"""
        X = np.array([p.to_vector() for p in players])  # Características
        y = np.array([p.performance for p in players])  # Etiquetas ("Bajo", etc.)
        self.model.fit(X, y)

    def predict(self, height, weight, avg_points):
        """Predice el rendimiento de un nuevo jugador"""
        data = np.array([[height, weight, avg_points]])
        return self.model.predict(data)[0]

    def evaluate(self, players):
        """Evalúa el modelo mostrando matriz de confusión e informe"""
        X = np.array([p.to_vector() for p in players])
        y_true = np.array([p.performance for p in players])
        y_pred = self.model.predict(X)

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


# ====================================================
# 4️⃣ Flujo principal: generar, entrenar, evaluar y visualizar
# ====================================================
class BasketballPredictionExample:
    def run(self):
        # 1️⃣ Generar los datos
        generator = BasketballDataGenerator(num_samples=200)
        players = generator.generate()

        # 2️⃣ Dividir en entrenamiento y prueba
        train_players, test_players = train_test_split(players, test_size=0.3, random_state=42)

        # 3️⃣ Entrenar el clasificador
        classifier = BasketballPerformanceClassifier()
        classifier.fit(train_players)

        # 4️⃣ Evaluar el modelo
        classifier.evaluate(test_players)

        # 5️⃣ Hacer una predicción para un nuevo jugador
        new_player = {"height": 198, "weight": 92, "avg_points": 17}
        predicted_perf = classifier.predict(**new_player)
        print(f"\n🎯 Predicción personalizada → Altura: {new_player['height']} cm, "
              f"Peso: {new_player['weight']} kg, Prom. puntos: {new_player['avg_points']}")
        print(f"   → Categoría predicha: {predicted_perf}")

        # 6️⃣ Visualizar los datos
        self.plot(players, new_player, predicted_perf)

    def plot(self, players, new_player, predicted_perf):
        """Gráfico de dispersión de altura vs puntos"""
        df = pd.DataFrame({
            "height": [p.height for p in players],
            "weight": [p.weight for p in players],
            "avg_points": [p.avg_points for p in players],
            "performance": [p.performance for p in players]
        })

        colors = {"Bajo": "red", "Medio": "orange", "Alto": "green"}

        plt.figure(figsize=(10, 6))
        for perf, color in colors.items():
            subset = df[df["performance"] == perf]
            plt.scatter(subset["height"], subset["avg_points"],
                        c=color, label=perf, alpha=0.7, edgecolors="k")

        # Nuevo jugador (estrella azul ⭐)
        plt.scatter(new_player["height"], new_player["avg_points"],
                    c="blue", s=200, marker="*", edgecolors="black",
                    label=f"Nuevo jugador ({predicted_perf})")

        plt.xlabel("Altura (cm)")
        plt.ylabel("Promedio de puntos por partido")
        plt.title("Clasificación de rendimiento de jugadores de baloncesto")
        plt.legend()
        plt.grid(True)
        plt.show()


# ====================================================
# 🚀 Ejecutar el ejemplo
# ====================================================
if __name__ == "__main__":
    example = BasketballPredictionExample()
    example.run()
