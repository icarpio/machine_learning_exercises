# ================================
#  Clase Player
# ================================

class Player:
    def __init__(self, player_name, character_type, avg_session_time, matches_played,
                 aggressive_actions, defensive_actions, items_bought, victories, style=None):

        self.player_name = player_name
        self.character_type = character_type
        self.avg_session_time = avg_session_time
        self.matches_played = matches_played
        self.aggressive_actions = aggressive_actions
        self.defensive_actions = defensive_actions
        self.items_bought = items_bought
        self.victories = victories
        self.style = style  # Puede ser None para predicciones

# ================================
#  Clase GameModel
# ================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

class GameModel:

    # Constructor que recibe una lista de jugadores
    def __init__(self, players_list):
        self.players = players_list

        # Codificadores para variables categóricas
        self.char_encoder = LabelEncoder()
        self.style_encoder = LabelEncoder()

        # Modelos
        self.class_model = None
        self.reg_model = None
        self.cluster_model = None

        # Almacén interno de DataFrame
        self.df = self._players_to_dataframe()

    # Convierte lista de Player en DataFrame
    def _players_to_dataframe(self):
        data = {
            "player_name": [p.player_name for p in self.players],
            "character_type": [p.character_type for p in self.players],
            "avg_session_time": [p.avg_session_time for p in self.players],
            "matches_played": [p.matches_played for p in self.players],
            "aggressive_actions": [p.aggressive_actions for p in self.players],
            "defensive_actions": [p.defensive_actions for p in self.players],
            "items_bought": [p.items_bought for p in self.players],
            "victories": [p.victories for p in self.players],
            "style": [p.style for p in self.players]
        }
        return pd.DataFrame(data)

    # -------------------------------
    #  Entrenamiento modelos
    # -------------------------------

    def train_classification_model(self):
        df = self.df.copy()

        # Codificar el tipo de personaje
        df["char_encoded"] = self.char_encoder.fit_transform(df["character_type"])
        df["style_encoded"] = self.style_encoder.fit_transform(df["style"])

        # Variables predictoras
        X = df[["char_encoded", "avg_session_time", "matches_played",
                "aggressive_actions", "defensive_actions", "items_bought"]]

        # Variable objetivo
        y = df["style_encoded"]

        # Modelo de clasificación
        self.class_model = RandomForestClassifier()
        self.class_model.fit(X, y)

    def train_regression_model(self):
        df = self.df.copy()
        df["char_encoded"] = self.char_encoder.transform(df["character_type"])

        X = df[["char_encoded", "avg_session_time", "matches_played",
                "aggressive_actions", "defensive_actions", "items_bought"]]

        y = df["victories"]

        # Modelo regresión
        self.reg_model = RandomForestRegressor()
        self.reg_model.fit(X, y)

    def train_clustering_model(self, n_clusters=2):
        df = self.df.copy()
        df["char_encoded"] = self.char_encoder.transform(df["character_type"])

        X = df[["char_encoded", "avg_session_time", "matches_played",
                "aggressive_actions", "defensive_actions", "items_bought"]]

        # KMeans
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_model.fit(X)

    # -------------------------------
    #  Métodos de predicción
    # -------------------------------

    def _player_to_model_input(self, player):
        """Convierte un Player en un vector de características."""
        char_encoded = self.char_encoder.transform([player.character_type])[0]

        return [[
            char_encoded,
            player.avg_session_time,
            player.matches_played,
            player.aggressive_actions,
            player.defensive_actions,
            player.items_bought
        ]]

    def predict_style(self, player):
        X = self._player_to_model_input(player)
        pred = self.class_model.predict(X)[0]
        return self.style_encoder.inverse_transform([pred])[0]

    def predict_victories(self, player):
        X = self._player_to_model_input(player)
        return self.reg_model.predict(X)[0]

    def assign_cluster(self, player):
        X = self._player_to_model_input(player)
        return int(self.cluster_model.predict(X)[0])

    # ================================
    #  OPCIONAL: Mostrar jugadores por cluster
    # ================================
    def show_players_by_cluster(self):
        df = self.df.copy()
        df["cluster"] = self.cluster_model.labels_

        # Mostrar jugadores agrupados
        for cluster_id in sorted(df["cluster"].unique()):
            print(f"\nCluster {cluster_id}:")
            cluster_players = df[df["cluster"] == cluster_id]

            for _, row in cluster_players.iterrows():
                print(f"{row['player_name']} - {row['character_type']} - {row['style']}")
                
                
# Crear datos de prueba para varios jugadores
players_data = [
    Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
    Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
    Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
    Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
]

# Instanciar modelo
model = GameModel(players_data)

# Entrenar modelos
model.train_classification_model()
model.train_regression_model()
model.train_clustering_model()

# Jugador de prueba
new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)

# Predicciones
predicted_style = model.predict_style(new_player)
predicted_victories = model.predict_victories(new_player)
predicted_cluster = model.assign_cluster(new_player)

# Resultados esperados
print(f"Estilo de juego predicho para {new_player.player_name}: {predicted_style}")
print(f"Victorias predichas para {new_player.player_name}: {predicted_victories:.2f}")
print(f"Cluster asignado a {new_player.player_name}: {predicted_cluster}")

# Mostrar clusters
model.show_players_by_cluster()

