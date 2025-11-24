""""
Predicción del consumo energético
📘 Contexto:

La eficiencia energética es una prioridad en las ciudades modernas. Las compañías eléctricas intentan predecir cuánto se consumirá en función de las condiciones meteorológicas. 
En este proyecto, desarrollarás un modelo de regresión lineal que permita predecir el consumo de energía en función de la temperatura ambiental.

🎯 Objetivo del proyecto:

Construir un sistema que:

Genere datos sintéticos con numpy representando temperatura (°C) y consumo energético (kWh).
Use regresión lineal (sklearn.linear_model.LinearRegression) para aprender la relación entre ambas variables.
Permita hacer predicciones para nuevas temperaturas.
Visualice los datos y el modelo con matplotlib.

🛠️ Requerimientos:

1. Crear una clase EnergyRecord

Guarda los atributos: temperature y consumption.

Añade un método .to_vector() que devuelva [temperature] como vector de entrada al modelo.

2. Generar los datos con una clase EnergyDataGenerator

Crea datos sintéticos con numpy.random.uniform(-5, 35) para la temperatura.

Calcula el consumo simulando que cuando hace más frío o más calor que 20 °C, el consumo aumenta:

consumo = 100 + (abs(temperatura - 20) * 3) + ruido
Añade un poco de ruido con numpy.random.normal(0, 5).
El método generate() devuelve una lista de objetos EnergyRecord

3. Crear la clase EnergyRegressor

Usa LinearRegression de sklearn para ajustar el modelo.

Métodos necesarios:

fit() para entrenar con una lista de EnergyRecord.
predict(temperature) para predecir consumo dado una temperatura.
get_model() para acceder al modelo (útil para graficar).

4. Implementar una clase EnergyPredictionExample

Que cree los datos, entrene el modelo y prediga para una temperatura nueva (por ejemplo, 30 °C).
También debe mostrar una gráfica:
Un scatter plot de los datos.
Una línea roja representando la recta de regresión.

5. Visualización con matplotlib

Agrega títulos, etiquetas de ejes y leyenda para una mejor comprensión.
Usa .plot() para la línea de predicción del modelo.

✅ Ejemplo de uso

example = EnergyPredictionExample()
example.run()


Salida esperada

🔍 Temperatura: 30 °C
⚡ Predicción de consumo: 120.70 kWh

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =====================================================================
# 1. Clase EnergyRecord
# =====================================================================

class EnergyRecord:
    """
    Representa un dato diario con temperatura y consumo eléctrico.
    Guarda la info del día.
    """
    def __init__(self, temperature, consumption):
        self.temperature = temperature      # grados Celsius
        self.consumption = consumption      # consumo eléctrico (MWh)

    def to_vector(self):
        """Devuelve la temperatura como vector para entrenar el modelo."""
        return [self.temperature]


# =====================================================================
# 2. Generador de datos REALISTAS de España
# =====================================================================

class SpainEnergyDataGenerator:
    """
    Genera temperaturas y consumos "realistas" parecidos a España.
    Todo es simulado, pero basado en rangos reales aproximados.
    """

    def generate(self, n_days=365):

        # Temperaturas mensuales aproximadas
        monthly_temp_ranges = [
            (7, 12),   # Ene
            (8, 13),   # Feb
            (10, 15),  # Mar
            (12, 18),  # Abr
            (14, 22),  # May
            (18, 28),  # Jun
            (20, 32),  # Jul
            (21, 31),  # Ago
            (18, 26),  # Sep
            (13, 21),  # Oct
            (9, 15),   # Nov
            (7, 12)    # Dic
        ]

        temperatures = []
        for month_range in monthly_temp_ranges:
            low, high = month_range
            days_in_month = int(n_days / 12)
            temps = np.random.uniform(low, high, days_in_month)
            temperatures.extend(temps)

        temperatures = np.array(temperatures[:n_days])

        # --- Simulación de consumo REALISTA ---
        # España consume ~250–350 GWh/día
        # más cuando hace calor extremo o frío extremo
        baseline = 250_000  # 250 GWh
        noise = np.random.normal(0, 5000, n_days)  # variación realista

        consumptions = baseline + np.abs(temperatures - 18) * 9000 + noise
        
        # Convertir a MWh (solo por claridad, ya están en MWh)
        return [EnergyRecord(t, c) for t, c in zip(temperatures, consumptions)]


# =====================================================================
# 3. Modelo de regresión
# =====================================================================

class EnergyRegressor:
    """
    Modelo simple de regresión lineal (una recta) para predecir consumo.
    """
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([rec.to_vector() for rec in records])
        y = np.array([rec.consumption for rec in records])
        self.model.fit(X, y)

    def predict(self, temperature):
        return float(self.model.predict([[temperature]]))

    def get_model(self):
        return self.model


# =====================================================================
# 4. Clase que lo ejecuta TODO
# =====================================================================

class EnergyPredictionExample:

    def __init__(self):
        self.generator = SpainEnergyDataGenerator()
        self.regressor = EnergyRegressor()

    def run(self):

        # Generar datos realistas simulados
        records = self.generator.generate()

        # Entrenar modelo
        self.regressor.fit(records)

        # Ejemplo de predicción
        #temp_nueva = 30
        temp_nueva = float(input("Introduce la temperatura (°C) para predecir el consumo: "))
        pred = self.regressor.predict(temp_nueva)

        print("===============================================")
        print("   🔥 PREDICCIÓN DE CONSUMO ELÉCTRICO (España)")
        print("===============================================")
        print(f"Temperatura simulada: {temp_nueva} °C")
        print(f"Consumo previsto: {pred:,.2f} MWh")
        print("===============================================")

        # Graficar
        self.plot(records)

    def plot(self, records):

        temps = [r.temperature for r in records]
        cons = [r.consumption for r in records]

        plt.scatter(temps, cons, s=15, alpha=0.6, label="Datos reales simulados")

        # Línea de regresión
        xs = np.linspace(min(temps), max(temps), 200)
        ys = self.regressor.get_model().predict(xs.reshape(-1, 1))

        plt.plot(xs, ys, color="red", linewidth=2, label="Modelo lineal")

        plt.xlabel("Temperatura (°C)")
        plt.ylabel("Consumo eléctrico diario (MWh)")
        plt.title("Relación Temp ↔ Consumo (España, datos simulados)")
        plt.legend()
        plt.grid(True)
        plt.show()


# =====================================================================
# 5. Ejecutar
# =====================================================================

if __name__ == "__main__":
    example = EnergyPredictionExample()
    example.run()
