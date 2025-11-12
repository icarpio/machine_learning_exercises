"""
Recomendador de videojuegos
🧠 Descripción:

Vas a construir un sistema de recomendación de videojuegos que pueda predecir si a un jugador le gustará o no un videojuego basándose en características como la acción, la estrategia, los gráficos o la dificultad.

Para ello, utilizarás:

Datos sintéticos generados con numpy
Un modelo de clasificación usando Random Forest de sklearn



🕹️ Objetivo:

Crear una clase VideoGame que represente un videojuego con características numéricas.
Generar una lista de videojuegos con etiquetas (le gusta/no le gusta) usando reglas sencillas.
Entrenar un modelo con RandomForestClassifier.
Usar el modelo para predecir si un nuevo videojuego será del gusto de un jugador.



🧩 Especificaciones del ejercicio:

Crea una clase VideoGame con los siguientes atributos:

action (nivel de acción, float de 0 a 1)
strategy (nivel de estrategia, float de 0 a 1)
graphics (calidad gráfica, float de 0 a 1)
difficulty (nivel de dificultad, float de 0 a 1)
liked (opcional: 1 si le gusta al jugador, 0 si no)

Crea una clase VideoGameGenerator que se se encargará de generar videojuegos aleatorios con sus características. Para ello:

Usa la función np.random.uniform(0, 1) para generar cada característica (action, strategy, graphics, difficulty).
Redondea los valores a dos decimales con np.round(..., 2).
Calcula automáticamente si al jugador le gustará el juego, usando una regla simple:
liked = int((action > 0.6 or graphics > 0.7) and difficulty < 0.7)
Cada objeto generado debe ser una instancia de la clase VideoGame, incluyendo su etiqueta liked.

Crea la clase VideoGameClassifier :

Entrene un modelo de clasificación usando RandomForestClassifier de scikit-learn, con el parámetro n_estimators=100, que indica que se usarán 100 árboles en el bosque aleatorio.
Pueda predecir si le gustará un nuevo videojuego al jugador, a partir de sus características numéricas (action, strategy, graphics, difficulty).
Crea una clase de ejemplo VideoGameRecommendationExample  donde:
Generas 100 videojuegos aleatorios con VideoGameGenerator.

Entrena un modelo con esos datos.

Crea un nuevo videojuego con las siguientes características:

new_game = VideoGame(action=0.9, strategy=0.4, graphics=0.8, difficulty=0.3)
Predice si le gustará al jugador.
Muestra por pantalla las características del juego y si se predice que gustará o no.
"""

# ----------------------------------------------------------
# 🧩 Importamos librerías necesarias
# ----------------------------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------
# 🎮 CLASE 1: VideoGame
# Representa un videojuego con sus características básicas.
# ----------------------------------------------------------
class VideoGame:
    def __init__(self, action, strategy, graphics, difficulty, liked=None):
        # Guardamos las características numéricas del juego (entre 0 y 1)
        self.action = action        # Nivel de acción (0 = nada, 1 = mucha acción)
        self.strategy = strategy    # Nivel de estrategia
        self.graphics = graphics    # Calidad gráfica
        self.difficulty = difficulty  # Dificultad
        self.liked = liked          # Si le gusta al jugador (1 = sí, 0 = no)


# ----------------------------------------------------------
# 🎲 CLASE 2: VideoGameGenerator
# Genera videojuegos aleatorios y calcula si gustarán o no.
# ----------------------------------------------------------
class VideoGameGenerator:
    def __init__(self, n_games=100, seed=42):
        np.random.seed(seed)  # Para resultados reproducibles
        self.n_games = n_games

    def generate_games(self):
        games = []
        for _ in range(self.n_games):
            # Generamos valores aleatorios entre 0 y 1 y redondeamos a 2 decimales
            action = np.round(np.random.uniform(0, 1), 2)
            strategy = np.round(np.random.uniform(0, 1), 2)
            graphics = np.round(np.random.uniform(0, 1), 2)
            difficulty = np.round(np.random.uniform(0, 1), 2)

            # Regla sencilla para decidir si gustará o no:
            # Si tiene buena acción o buenos gráficos y no es muy difícil → gusta
            liked = int((action > 0.6 or graphics > 0.7) and difficulty < 0.7)

            # Creamos un objeto VideoGame y lo añadimos a la lista
            games.append(VideoGame(action, strategy, graphics, difficulty, liked))
        return games


# ----------------------------------------------------------
# 🧠 CLASE 3: VideoGameClassifier
# Entrena un modelo Random Forest y predice si gustará o no.
# ----------------------------------------------------------
class VideoGameClassifier:
    def __init__(self):
        # Creamos el modelo Random Forest con 100 árboles
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, games):
        # Convertimos la lista de objetos VideoGame a arrays NumPy
        X = np.array([[g.action, g.strategy, g.graphics, g.difficulty] for g in games])
        y = np.array([g.liked for g in games])

        # Entrenamos el modelo
        self.model.fit(X, y)
        print(f"✅ Modelo entrenado con {len(games)} videojuegos.")

    def predict(self, new_game):
        # Creamos el vector de características del nuevo videojuego
        features = np.array([[new_game.action, new_game.strategy,
                              new_game.graphics, new_game.difficulty]])
        # Hacemos la predicción
        prediction = self.model.predict(features)[0]
        return prediction


# ----------------------------------------------------------
# 🚀 CLASE 4: VideoGameRecommendationExample
# Crea el flujo completo: generar datos, entrenar y predecir.
# ----------------------------------------------------------
class VideoGameRecommendationExample:
    def run(self):
        print("\n🎮 Bienvenido al recomendador de videojuegos 🎮")
        print("Generando videojuegos y entrenando modelo...")

        # 1️⃣ Generamos videojuegos de ejemplo
        generator = VideoGameGenerator(n_games=100)
        games = generator.generate_games()

        # 2️⃣ Entrenamos el modelo
        classifier = VideoGameClassifier()
        classifier.train(games)

        # 3️⃣ Creamos un nuevo videojuego manualmente
        new_game = VideoGame(action=0.9, strategy=0.4, graphics=0.8, difficulty=0.3)

        # 4️⃣ Mostramos sus características
        print("\n🎮 Nuevo juego:")
        print(f"Action: {new_game.action}, Strategy: {new_game.strategy}, "
              f"Graphics: {new_game.graphics}, Difficulty: {new_game.difficulty}")

        # 5️⃣ Predicción
        result = classifier.predict(new_game)

        # 6️⃣ Mostramos resultado final
        if result == 1:
            print("✅ ¿Le gustará al jugador el juego? ¡Sí!")
        else:
            print("❌ ¿Le gustará al jugador el juego? No.")


# ----------------------------------------------------------
# 🧪 EJECUCIÓN DEL EJEMPLO
# ----------------------------------------------------------
if __name__ == "__main__":
    example = VideoGameRecommendationExample()
    example.run()
