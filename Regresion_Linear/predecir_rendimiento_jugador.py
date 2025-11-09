# Importamos el modelo de regresión lineal de scikit-learn
from sklearn.linear_model import LinearRegression


# ---------------------------
# 1️⃣ Clase Player
# ---------------------------
class Player:
    """
    Representa a un jugador del videojuego.
    Contiene sus estadísticas promedio y (opcionalmente) su número de victorias.
    """
    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=None):
        # Guardamos los atributos
        self.name = name
        self.avg_session_time = avg_session_time           # Duración promedio de sesión (minutos)
        self.avg_actions_per_min = avg_actions_per_min     # Acciones por minuto
        self.avg_kills_per_session = avg_kills_per_session # Eliminaciones por sesión
        self.victories = victories                         # Victorias (opcional, no siempre se conoce)

    def to_features(self):
        """
        Devuelve las características que el modelo usará como entrada.
        (En Machine Learning, esto se conoce como vector de características o 'features')
        """
        return [self.avg_session_time, self.avg_actions_per_min, self.avg_kills_per_session]



# ---------------------------
# 2️⃣ Clase PlayerDataset
# ---------------------------
class PlayerDataset:
    """
    Representa un conjunto de jugadores (nuestro conjunto de datos de entrenamiento).
    Proporciona métodos para obtener las matrices necesarias para entrenar el modelo.
    """
    def __init__(self, players):
        self.players = players  # Lista de objetos Player

    def get_feature_matrix(self):
        """
        Devuelve una lista de listas (matriz X) con las características de cada jugador.
        Ejemplo:
        [
          [40, 50, 6],
          [30, 35, 4],
          [50, 60, 7],
          ...
        ]
        """
        return [p.to_features() for p in self.players if p.victories is not None]

    def get_target_vector(self):
        """
        Devuelve la lista de valores objetivo (y), es decir, las victorias reales.
        Ejemplo:
        [20, 10, 25, 5, 30]
        """
        return [p.victories for p in self.players if p.victories is not None]

# ---------------------------
# 3️⃣ Clase VictoryPredictor
# ---------------------------
class VictoryPredictor:
    """
    Clase encargada de entrenar el modelo de regresión y hacer predicciones.
    """
    def __init__(self):
        # Creamos el modelo de regresión lineal
        self.model = LinearRegression()

    def train(self, dataset: PlayerDataset):
        """
        Entrena el modelo usando los datos del dataset.
        """
        X = dataset.get_feature_matrix()  # Características
        y = dataset.get_target_vector()   # Victorias
        self.model.fit(X, y)              # Entrenamiento (aquí el modelo aprende los coeficientes)

    def predict(self, player: Player):
        """
        Predice las victorias para un jugador nuevo (sin dato de victorias).
        """
        X_test = [player.to_features()]     # Convertimos al formato correcto (lista dentro de lista)
        return self.model.predict(X_test)[0]  # Devolvemos el número de victorias predicho


# ---------------------------
# 🧪 Ejemplo de uso
# ---------------------------
if __name__ == "__main__":
    # Creamos algunos jugadores conocidos (datos de entrenamiento)
    players = [
        Player("Alice", 40, 50, 6, 20),
        Player("Bob", 30, 35, 4, 10),
        Player("Charlie", 50, 60, 7, 25),
        Player("Diana", 20, 25, 2, 5),
        Player("Eve", 60, 70, 8, 30)
    ]

    # Creamos el dataset y entrenamos el modelo
    dataset = PlayerDataset(players)
    predictor = VictoryPredictor()
    predictor.train(dataset)

    # Jugador nuevo del que queremos predecir las victorias
    test_player = Player("TestPlayer", 45, 55, 5)

    # Realizamos la predicción
    predicted = predictor.predict(test_player)
    print(f"Victorias predichas para {test_player.name}: {predicted:.2f}")
