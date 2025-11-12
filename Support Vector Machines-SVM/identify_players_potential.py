"""
Detectar jugadores con potencial profesional
🎮 Ejercicio: ¿Quién será un jugador profesional? - Clasificación con SVM

🧠 Contexto:

Imagina que trabajas en una plataforma de eSports y tu equipo está desarrollando un sistema de scouting para detectar jugadores 
con potencial profesional en base a sus estadísticas de juego.

Tu tarea es construir un modelo de clasificación usando SVM (Support Vector Machine) que, dada la información de un jugador, 
prediga si tiene el perfil de jugador profesional (1) o casual (0).

📋 Objetivo del ejercicio

Implementar un clasificador que:

Use datos simulados de jugadores (partidas ganadas, horas jugadas, precisión, velocidad de reacción, estrategia).

Entrene un modelo de SVM con scikit-learn.
Clasifique a nuevos jugadores como “profesional” o “casual”.
Evalúe el rendimiento del modelo utilizando la métrica de precisión (accuracy_score).

📁 Datos de entrada:

Cada jugador se representa con las siguientes características (todas normalizadas entre 0 y 1):

🧪 Ejemplo de datos

simulator = GameSimulator()
simulator.run()
Salida esperada

Jugador profesional:
Precisión del modelo: 1.0

🛠️ Tareas a realizar:

Implementa la clase ProPlayerClassifier con los métodos:

train(X, y) para entrenar el modelo.
predict(player_stats) para predecir si un jugador es profesional
evaluate(X_test, y_test): evalúa el modelo con precisión.
Usa sklearn.svm.SVC como modelo base.
Prueba el modelo con al menos dos predicciones de distintos jugadores.
Evalúa su rendimiento con accuracy_score de sklearn.metrics

"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ===================================================
# Simulador de datos de jugadores
# ===================================================
class GameSimulator:
    def __init__(self, n_samples=200, random_state=42):
        np.random.seed(random_state)
        self.n_samples = n_samples

    def run(self):
        # Características: [partidas_ganadas, horas_jugadas, precisión, velocidad_reacción, estrategia]
        # Se normalizan con valores entre 0 y 1
        X = np.random.rand(self.n_samples, 5)
        
        """
        Esto calcula una especie de “puntaje total del jugador”, donde cada característica tiene una importancia distinta:

        🏆 partidas_ganadas (30%) — muy importante
        ⏰ horas_jugadas (20%) — relevante, pero menos
        🎯 precisión (25%) — también clave
        ⚡ velocidad de reacción (15%) — importante pero secundaria
        🧩 estrategia (10%) — algo menos influyente

        En resumen:

        Es una fórmula “inventada” para que el modelo tenga una lógica que imite el rendimiento de un jugador.
        Regla simulada: jugadores profesionales tienen mejor desempeño global
        
        umbral > 0.6  --> Si el puntaje total es mayor a 0.6, consideramos que el jugador tiene perfil profesional (1).
        
        .astype(int) Convierte el resultado booleano (True/False) en valores enteros (1/0).
        """
        y = (0.3*X[:,0] + 0.2*X[:,1] + 0.25*X[:,2] + 0.15*X[:,3] + 0.1*X[:,4] > 0.6).astype(int)
        return X, y

# ===================================================
# Clasificador con SVM
# ===================================================
class ProPlayerClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', gamma='scale', C=1.0)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, player_stats):
        player_stats = np.array(player_stats).reshape(1, -1)
        prediction = self.model.predict(player_stats)
        return int(prediction[0])  # ✅ devuelve 0 o 1

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

# ===================================================
# Ejecución de ejemplo
# ===================================================
if __name__ == "__main__":
    simulator = GameSimulator()
    X, y = simulator.run()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = ProPlayerClassifier()
    clf.train(X_train, y_train)

    acc = clf.evaluate(X_test, y_test)
    print(f"Precisión del modelo: {acc:.2f}")
    
    """
    # ===========================
    # Entrada manual de 3 jugadores
    # ===========================
    num_jugadores = 3
    print(f"\nIntroduce las estadísticas de {num_jugadores} jugadores (valores entre 0 y 1):")

    for i in range(num_jugadores):
        print(f"\n👤 Jugador {i+1}")
        partidas_ganadas = float(input("Partidas ganadas: "))
        horas_jugadas = float(input("Horas jugadas: "))
        precision = float(input("Precisión: "))
        velocidad = float(input("Velocidad de reacción: "))
        estrategia = float(input("Estrategia: "))

        jugador = [partidas_ganadas, horas_jugadas, precision, velocidad, estrategia]

        resultado = clf.predict(jugador)
        if resultado == 1:
            print("🏆 El jugador tiene perfil PROFESIONAL")
        else:
            print("🎮 El jugador tiene perfil CASUAL")
    """
    # Ejemplo de predicciones
    jugador1 = [0.9, 0.8, 0.95, 0.85, 0.9]  # debería ser profesional (1)
    jugador2 = [0.2, 0.3, 0.4, 0.3, 0.25]  # debería ser casual (0)

    print("Jugador 1:", clf.predict(jugador1))
    print("Jugador 2:", clf.predict(jugador2))
