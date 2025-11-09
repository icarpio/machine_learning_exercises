"""
Recomendador de Personajes
🎮 "Recomendador de Personajes: ¿Qué tipo de personaje deberías elegir?"

📘 Enunciado

En este ejercicio trabajarás como desarrollador de sistemas inteligentes para un nuevo videojuego tipo RPG online. 
El juego permite a los jugadores crear personajes y elegir entre distintos roles o clases (por ejemplo: guerrero, mago, arquero, curandero…).

Tu tarea es construir un modelo de recomendación que, dado un perfil de jugador (nivel, estilo de combate, número de partidas jugadas, etc.), 
recomiende qué tipo de personaje debería usar, basándose en datos históricos de otros jugadores similares.

🧩 Requerimientos

Crea una clase Player que represente a un jugador con los siguientes atributos:

name: nombre del jugador.
level: nivel del jugador (1 a 100).
aggressiveness: valor entre 0 y 1 que representa su estilo ofensivo.
cooperation: valor entre 0 y 1 que representa cuánto coopera con el equipo.
exploration: valor entre 0 y 1 que representa cuánto le gusta explorar el mapa.
preferred_class: clase de personaje que suele elegir (solo en los datos de entrenamiento).
Implementa un método .to_features() en la clase para convertir al jugador en una lista de características numéricas (sin la clase preferida).

Crea una clase PlayerDataset que contenga una lista de jugadores y proporcione:

get_X() → lista de listas de características.
get_y() → lista de clases preferidas.

Crea una clase ClassRecommender que use KNN para:

Entrenar el modelo a partir de un PlayerDataset.
Predecir la mejor clase para un nuevo jugador (predict(player)).
Obtener los k jugadores más parecidos (get_nearest_neighbors(player)).
(Opcional) Permite probar diferentes valores de k y evaluar la precisión del modelo con cross_val_score.

"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# -----------------------------
# Clase Player
# -----------------------------
class Player:
    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        self.name = name
        self.level = level
        self.aggressiveness = aggressiveness
        self.cooperation = cooperation
        self.exploration = exploration
        self.preferred_class = preferred_class

    def to_features(self):
        """Convierte el jugador en una lista de características numéricas."""
        return [self.level, self.aggressiveness, self.cooperation, self.exploration]


# -----------------------------
# Clase PlayerDataset
# -----------------------------
class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_X(self):
        """Devuelve las características (X) de todos los jugadores."""
        return [p.to_features() for p in self.players]

    def get_y(self):
        """Devuelve las clases preferidas (y) de los jugadores."""
        return [p.preferred_class for p in self.players]


# -----------------------------
# Clase ClassRecommender (KNN)
# -----------------------------
class ClassRecommender:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train(self, dataset: PlayerDataset):
        """Entrena el modelo KNN usando los datos del dataset."""
        X = dataset.get_X()
        y = dataset.get_y()
        self.model.fit(X, y)

    def predict(self, player: Player):
        """Predice la clase recomendada para un nuevo jugador."""
        return self.model.predict([player.to_features()])[0]

    def get_nearest_neighbors(self, player: Player):
        """Devuelve los índices de los jugadores más parecidos al jugador dado."""
        distances, indices = self.model.kneighbors([player.to_features()])
        return indices[0]

    def evaluate(self, dataset: PlayerDataset, cv=5):
        """(Opcional) Evalúa la precisión media del modelo con validación cruzada."""
        X = dataset.get_X()
        y = dataset.get_y()
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores.mean()


# -----------------------------
# 🧪 Ejemplo de uso
# -----------------------------
if __name__ == "__main__":
    # Datos de entrenamiento
    players = [
        Player("Alice", 20, 0.8, 0.2, 0.1, "Warrior"),
        Player("Bob", 45, 0.4, 0.8, 0.2, "Healer"),
        Player("Cleo", 33, 0.6, 0.4, 0.6, "Archer"),
        Player("Dan", 60, 0.3, 0.9, 0.3, "Healer"),
        Player("Eli", 50, 0.7, 0.2, 0.9, "Mage"),
        Player("Fay", 25, 0.9, 0.1, 0.2, "Warrior"),
    ]

    # Nuevo jugador
    new_player = Player("TestPlayer", 40, 0.6, 0.3, 0.8)

    # Entrenamiento y predicción
    dataset = PlayerDataset(players)
    recommender = ClassRecommender(n_neighbors=3)
    recommender.train(dataset)

    recommended_class = recommender.predict(new_player)
    neighbors_indices = recommender.get_nearest_neighbors(new_player)

    # Resultados
    print(f"Clase recomendada para {new_player.name}: {recommended_class}")
    print("Jugadores similares:")
    for i in neighbors_indices:
        print(f"- {players[i].name} ({players[i].preferred_class})")
