# Importamos las librerías necesarias
import numpy as np                        # NumPy: librería para manejar arreglos y generar datos aleatorios numéricos
from sklearn.linear_model import LogisticRegression   # De scikit-learn: modelo de regresión logística

# 📦 Paso 1: Definimos una clase que representa los datos de un jugador en una partida
class PlayerMatchData:
    def __init__(self, kills, deaths, assists, damage_dealt, damage_received, healing_done, objective_time, won):
        # Guardamos cada variable como un atributo del objeto
        self.kills = kills                      # Número de enemigos eliminados
        self.deaths = deaths                    # Número de veces que murió
        self.assists = assists                  # Número de asistencias
        self.damage_dealt = damage_dealt        # Daño total infligido
        self.damage_received = damage_received  # Daño total recibido
        self.healing_done = healing_done        # Curación total realizada
        self.objective_time = objective_time    # Tiempo capturando objetivos (en segundos)
        self.won = won                          # Resultado: 1 si ganó, 0 si perdió

    # Método que devuelve los datos en forma de diccionario (útil para visualización o conversión)
    def to_dict(self, include_won=False):
        data = {
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "damage_dealt": self.damage_dealt,
            "damage_received": self.damage_received,
            "healing_done": self.healing_done,
            "objective_time": self.objective_time
        }
        # Si el usuario quiere incluir el resultado (ganó o no), lo añadimos
        if include_won:
            data["won"] = self.won
        return data


# 📦 Paso 2: Generamos datos sintéticos (ficticios) para entrenar el modelo
def generate_synthetic_data(n=100):
    data = []  # Aquí guardaremos todos los objetos PlayerMatchData creados
    
    # Generamos 'n' jugadores simulados
    for _ in range(n):
        # np.random.poisson(media): genera un número aleatorio siguiendo una distribución Poisson
        # Esta distribución es útil para contar eventos (como kills o muertes)
        kills = np.random.poisson(5)   # En promedio 5 enemigos eliminados
        deaths = np.random.poisson(3)  # En promedio 3 muertes
        assists = np.random.poisson(2) # En promedio 2 asistencias

        # Generamos el daño infligido y recibido
        # Usamos ruido normal (np.random.normal) para hacerlo más realista
        damage_dealt = kills * 300 + np.random.normal(0, 100)       # Cada kill aporta unos 300 puntos de daño
        damage_received = deaths * 400 + np.random.normal(0, 100)   # Cada muerte implica recibir unos 400 puntos de daño

        # Generamos curación y tiempo en objetivo de forma aleatoria
        healing_done = np.random.randint(0, 301)  # Valor entre 0 y 300
        objective_time = np.random.randint(0, 121) # Valor entre 0 y 120 segundos

        # Definimos la lógica del resultado:
        # El jugador gana si infligió más daño del que recibió y tuvo más kills que muertes
        won = 1 if (damage_dealt > damage_received and kills > deaths) else 0

        # Creamos un objeto PlayerMatchData con todos estos valores
        player = PlayerMatchData(
            kills, deaths, assists,
            damage_dealt, damage_received,
            healing_done, objective_time, won
        )

        # Agregamos el jugador a la lista de datos
        data.append(player)
    
    # Devolvemos la lista completa con todos los jugadores simulados
    return data


# 🧠 Paso 3: Creamos una clase que entrena un modelo de Machine Learning
class VictoryPredictor:
    def __init__(self):
        # LogisticRegression: modelo supervisado de clasificación binaria
        # (usa una función sigmoide para predecir probabilidades entre 0 y 1)
        self.model = LogisticRegression()

    # Método para entrenar el modelo con datos
    def train(self, data):
        # Creamos la matriz de características (X)
        # Cada fila representa un jugador y cada columna una estadística
        X = np.array([
            [
                d.kills, d.deaths, d.assists,
                d.damage_dealt, d.damage_received,
                d.healing_done, d.objective_time
            ]
            for d in data
        ])
        
        # Creamos el vector de etiquetas (y)
        # 1 = ganó, 0 = perdió
        y = np.array([d.won for d in data])
        
        # Entrenamos el modelo con los datos
        self.model.fit(X, y)

    # Método para predecir el resultado de un nuevo jugador
    def predict(self, player: PlayerMatchData):
        # Creamos una fila con los datos del nuevo jugador
        X_new = np.array([[
            player.kills, player.deaths, player.assists,
            player.damage_dealt, player.damage_received,
            player.healing_done, player.objective_time
        ]])
        
        # Usamos el modelo entrenado para predecir (devuelve [0] o [1])
        return int(self.model.predict(X_new)[0])


# 📊 Ejemplo de uso del modelo
if __name__ == "__main__":
    # Generamos 150 partidas simuladas para entrenar el modelo
    training_data = generate_synthetic_data(150)
    
    # Creamos el predictor y lo entrenamos con los datos generados
    predictor = VictoryPredictor()
    predictor.train(training_data)
    
    # Creamos un nuevo jugador para probar el modelo
    test_player = PlayerMatchData(
        8,   # kills
        2,   # deaths
        3,   # assists
        2400,# damage_dealt
        800, # damage_received
        120, # healing_done
        90,  # objective_time
        None # No sabemos si ganó o no (lo queremos predecir)
    )
    
    # Obtenemos la predicción (1 = victoria, 0 = derrota)
    prediction = predictor.predict(test_player)
    
    # Mostramos el resultado en pantalla
    print(f"¿El jugador ganará? {'Sí' if prediction == 1 else 'No'}")

