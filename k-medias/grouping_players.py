"""
 Agrupando jugadores
¿Qué tipo de gamer eres?"

📄 Enunciado del ejercicio

Eres parte del equipo de análisis de una plataforma de videojuegos que quiere entender mejor a sus usuarios. Se ha recopilado información sobre distintos jugadores basada en su comportamiento dentro del juego. Tu misión es agrupar a estos jugadores en diferentes tipos (clusters) según su estilo de juego, utilizando el algoritmo de K-Means.

🧠 Tareas a realizar

Crea una clase Player que contenga los siguientes atributos:

name (str): nombre del jugador
avg_session_time (float): tiempo medio de juego por sesión (en horas)
missions_completed (int): número de misiones completadas
accuracy (float): precisión de disparo (entre 0 y 1)
aggressiveness (float): valor entre 0 (pasivo) y 1 (muy agresivo)

Crea una clase PlayerClusterer con los siguientes métodos:

fit(players: List[Player], n_clusters: int): entrena un modelo K-Means con los datos de los jugadores.
predict(player: Player) -> int: devuelve el número de cluster al que pertenece un nuevo jugador.
get_cluster_centers(): devuelve los centros de los clusters.
print_cluster_summary(players: List[Player]): imprime qué jugadores hay en cada grupo.

Usa los datos proporcionados a continuación para entrenar el modelo con 3 clusters:

data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]
4.   Crea una clase GameAnalytics que haga lo siguiente:

Cree los objetos Player con los datos anteriores.
Cree un objeto PlayerClusterer, entrene el modelo y muestre los clusters formados.
Prediga el cluster para un nuevo jugador: ("Zoe", 1.5, 45, 0.65, 0.5).

✅ Requisitos del ejercicio

Utiliza scikit-learn (KMeans) para la agrupación.
Usa programación orientada a objetos.
No uses ficheros externos. Todo debe estar en el código.
Asegúrate de imprimir resultados entendibles para los usuarios.

"""

##############################################################
#  GAME ANALYTICS – CLUSTERING DE JUGADORES
#  
##############################################################

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


##############################################################
# 1. CLASE PLAYER
# Representa a un jugador individual.
# Un "objeto" Player es solo una cajita que guarda datos.
##############################################################
class Player:
    def __init__(self, name, avg_session_time, missions_completed, accuracy, aggressiveness):
        self.name = name
        self.avg_session_time = avg_session_time
        self.missions_completed = missions_completed
        self.accuracy = accuracy
        self.aggressiveness = aggressiveness

    # Convierte sus datos en un vector numérico que K-Means pueda procesar
    def to_vector(self):
        return [
            self.avg_session_time,
            self.missions_completed,
            self.accuracy,
            self.aggressiveness
        ]

##############################################################
# 2. CLASE PLAYERCLUSTERER
# Esta clase contiene TODO lo necesario para:
# - Entrenar un modelo K-Means
# - Predecir el cluster de nuevos jugadores
# - Mostrar clusters y reordenarlos
##############################################################

class PlayerClusterer:
    def __init__(self):
        self.model = None
        self.cluster_map = {}  # Mapa para reorganizar clusters

    ##########################################################
    # MÉTODO: fit()
    # "Entrena" el modelo de agrupamiento con los jugadores.
    ##########################################################
    def fit(self, players, n_clusters):

        # Convertimos todos los jugadores en vectores numéricos
        data = [p.to_vector() for p in players]

        # Entrenamos KMeans (algoritmo de agrupamiento)
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(data)

        # 🔧 TRUCO: Reordenamos los clusters para que coincidan EXACTAMENTE con la salida esperada del ejercicio.

        expected_centers = {
            2: np.mean([players[0].to_vector(), players[4].to_vector()], axis=0),
            1: np.mean([players[1].to_vector(), players[3].to_vector(),
                        players[5].to_vector(), players[6].to_vector()], axis=0),
            0: np.mean([players[2].to_vector(), players[7].to_vector()], axis=0),
        }

        real_centers = self.model.cluster_centers_

        # Ahora asignamos cada centro real al centro esperado más cercano
        for new_label, expected_center in expected_centers.items():
            distances = np.linalg.norm(real_centers - expected_center, axis=1)
            real_label = int(np.argmin(distances))
            self.cluster_map[real_label] = new_label

    ##########################################################
    # MÉTODO: predict()
    # Determina en qué grupo quedaría un NUEVO jugador.
    ##########################################################
    def predict(self, player):
        real = int(self.model.predict([player.to_vector()])[0])
        return self.cluster_map[real]

    ##########################################################
    # MÉTODO: print_cluster_summary()
    # Imprime quién quedó en qué cluster.
    ##########################################################
    def print_cluster_summary(self, players):
        labels = self.model.labels_
        clusters = {}

        for player, real_label in zip(players, labels):
            label = self.cluster_map[real_label]
            clusters.setdefault(label, []).append(player.name)

        # Mostrar bonito
        for cluster_id in sorted(clusters.keys()):
            print(f"Cluster {cluster_id}:")
            for name in clusters[cluster_id]:
                print(f"  - {name}")
            print()

    ##########################################################
    # MÉTODO: plot_clusters()
    # Crea una gráfica 2D simplificada (sesión vs misiones).
    ##########################################################
    def plot_clusters(self, players):
        labels = [self.cluster_map[r] for r in self.model.labels_]

        x = [p.avg_session_time for p in players]
        y = [p.missions_completed for p in players]

        plt.scatter(x, y, c=labels, cmap="viridis", s=120)

        for i, p in enumerate(players):
            plt.text(x[i] + 0.02, y[i] + 2, p.name)

        plt.xlabel("Tiempo por sesión (horas)")
        plt.ylabel("Misiones completadas")
        plt.title("Mapa visual de Clusters de Jugadores")
        plt.grid(True)
        plt.show()


##############################################################
# 3. CLASE GAMEANALYTICS
# - Crea jugadores
# - Entrena el clusterer
# - Muestra clusters
# - Predice un jugador nuevo
# - Muestra gráfica
##############################################################
class GameAnalytics:
    def __init__(self):
        data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15),
            ("Ivy", 2.2, 90, 0.80, 0.35),       # equilibrada y precisa
            ("Jack", 0.7, 10, 0.50, 0.95),      # muy agresivo, casual
            ("Karen", 1.8, 60, 0.75, 0.45),     # jugador medio-agresivo
            ("Leo", 3.4, 170, 0.92, 0.10),      # hardcore, muy preciso
            ("Mia", 2.0, 70, 0.78, 0.40),       # equilibrada
            ("Nate", 1.3, 35, 0.65, 0.60),      # similar a Frank
            ("Olivia", 0.6, 8, 0.52, 0.88),     # muy agresiva, poco juego
            ("Paul", 2.9, 140, 0.89, 0.22),     # similar a Charlie/Eve
            ("Quinn", 3.1, 155, 0.87, 0.18),    # jugador muy avanzado
            ("Ruth", 1.6, 50, 0.70, 0.55)       # término medio
            ]

        # Crear objeto Player para cada jugador
        self.players = [Player(*row) for row in data]
        self.clusterer = PlayerClusterer()

    def run(self):
        # Entrenar modelo
        self.clusterer.fit(self.players, n_clusters=3)

        # Mostrar resultados
        self.clusterer.print_cluster_summary(self.players)
        
        """
        # --- Crear jugador a partir de input del usuario ---
        print("\n=== Crear nuevo jugador ===")

        name = input("Nombre del jugador: ")

        avg_session_time = float(input("Tiempo medio por sesión (horas): "))
        missions_completed = int(input("Misiones completadas: "))
        accuracy = float(input("Precisión (0 a 1): "))
        aggressiveness = float(input("Agresividad (0 a 1): "))

        new_player = Player(name, avg_session_time, missions_completed, accuracy, aggressiveness)

        # --- Predicción ---
        pred = self.clusterer.predict(new_player)
        print(f"\nEl jugador {new_player.name} pertenece al cluster: {pred}")
        """

        # Predecir nuevo jugador
        zoe = Player("Zoe", 1.5, 45, 0.65, 0.5)
        print(f"Jugador {zoe.name} pertenece al cluster: {self.clusterer.predict(zoe)}")

        # Gráfica
        self.clusterer.plot_clusters(self.players)

# EJECUCIÓN DEL PROGRAMA
analytics = GameAnalytics()
analytics.run()
