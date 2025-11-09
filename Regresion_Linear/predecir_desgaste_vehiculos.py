"""
Predicción del desgaste de vehículos
🛠️ Misión: Predicción del Desgaste de Vehículos Militares

Como analista de datos en una base militar, tu tarea es predecir el nivel de desgaste de vehículos en función de las horas de uso.
Esta herramienta será clave para evitar fallos operativos y optimizar los mantenimientos preventivos.



🎯 Objetivo

Implementa un sistema basado en regresión lineal que:
Genere registros de entrenamiento con la clase VehicleDataGenerator.
Represente cada registro con la clase VehicleRecord, que almacene:

hours_used: horas de servicio del vehículo.
wear_level: nivel de desgaste en porcentaje (0 a 100).

Entrene un modelo de regresión lineal con la clase VehicleWearRegressor, que:
Reciba una lista de registros.
Aprenda la relación entre horas de uso y desgaste.
Permita hacer predicciones para nuevas horas.
Visualice los datos y prediga el desgaste con la clase VehicleWearPredictionExample.



🧱 Organización en clases



🚗 VehicleRecord
Representa un registro individual del uso de un vehículo.

Atributos:
hours_used: horas de servicio del vehículo.
wear_level: nivel de desgaste en porcentaje (0 a 100).

Método:
to_vector() → Devuelve una lista o vector con [hours_used] para usarlo en el modelo.

🧩 Pista:
Esta clase funciona como una “fila” de datos, que luego será usada para entrenar el modelo.



🧮 VehicleDataGenerator

Genera un conjunto de datos sintéticos (simulados) para entrenamiento.

Atributo:
num_samples: cantidad de registros a generar.

Método:

generate() → Devuelve una lista de objetos VehicleRecord.

El desgaste debe aumentar con las horas de uso, pero con cierta variabilidad aleatoria.
Puedes usar:

np.random.uniform() → para generar horas aleatorias entre 50 y 500.
np.random.normal() → para añadir ruido (variación natural).
np.clip() → para limitar el desgaste entre 0 y 100.

🧩 Pista:

Una posible relación lineal podría ser algo así:
desgaste = 10 + 0.18 * horas + ruido_aleatorio



🧰 VehicleWearRegressor
Entrena un modelo de regresión lineal que relacione horas de uso y desgaste.

Métodos:

fit(records): recibe una lista de VehicleRecord y entrena el modelo.
Convierte los datos en arrays X y y usando numpy.
predict(hours): devuelve el desgaste estimado para un número dado de horas.
get_model(): devuelve el modelo entrenado (LinearRegression).



📊 VehicleWearPredictionExample

Clase principal para ejecutar la simulación completa.

Método:
run(), que debe:
Generar los datos con VehicleDataGenerator.
Entrenar el modelo con VehicleWearRegressor.
Predecir el desgaste para un vehículo con 250 horas de uso.
Mostrar los resultados por pantalla.
Visualizar la relación con un gráfico:
Puntos verdes → Datos reales.
Línea roja → Línea de regresión.
Línea gris vertical → Nuevo vehículo.

🧩 Pista visual:
Usa matplotlib.pyplot para graficar los puntos (plt.scatter) y la línea (plt.plot).



⚙️ Tecnologías a usar
NumPy, Pandas, Matplotlib
LinearRegression de scikit-learn



🧪 Ejemplo de uso
example = VehicleWearPredictionExample()
example.run()

Salida esperada

⏱ Horas de uso estimadas: 250
⚙️ Nivel de desgaste estimado: 53.75%

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 🚗 Clase que representa un registro individual
class VehicleRecord:
    def __init__(self, hours_used: float, wear_level: float):
        self.hours_used = hours_used
        self.wear_level = wear_level

    def to_vector(self):
        """Devuelve un vector con las horas de uso (para el modelo)."""
        return [self.hours_used]


# 🧮 Generador de datos sintéticos
class VehicleDataGenerator:
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples

    def generate(self):
        """Genera una lista de objetos VehicleRecord con datos simulados."""
        records = []

        # Horas entre 50 y 500
        hours = np.random.uniform(50, 500, self.num_samples)

        # Desgaste = 10 + 0.18 * horas + ruido
        noise = np.random.normal(0, 5, self.num_samples)
        wear = 10 + 0.18 * hours + noise

        # Limitamos entre 0 y 100
        wear = np.clip(wear, 0, 100)

        for h, w in zip(hours, wear):
            records.append(VehicleRecord(h, w))

        return records


# 🧰 Modelo de regresión lineal
class VehicleWearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        """Entrena el modelo usando los registros de vehículos."""
        X = np.array([r.to_vector() for r in records])  # horas
        y = np.array([r.wear_level for r in records])   # desgaste

        self.model.fit(X, y)

    def predict(self, hours):
        """Predice el desgaste para un número de horas dado."""
        return float(self.model.predict(np.array([[hours]]))[0])

    def get_model(self):
        """Devuelve el modelo entrenado."""
        return self.model


# 📊 Ejemplo completo de predicción y visualización
class VehicleWearPredictionExample:
    def run(self):
        # 1️⃣ Generar datos de entrenamiento
        generator = VehicleDataGenerator(num_samples=100)
        records = generator.generate()

        # 2️⃣ Entrenar el modelo
        regressor = VehicleWearRegressor()
        regressor.fit(records)

        # 3️⃣ Predecir para 250 horas
        hours_to_predict = 250
        predicted_wear = regressor.predict(hours_to_predict)

        # 4️⃣ Mostrar resultados
        print("⏱ Horas de uso estimadas:", hours_to_predict)
        print(f"⚙️ Nivel de desgaste estimado: {predicted_wear:.2f}%")

        # 5️⃣ Visualizar resultados
        X = np.array([r.hours_used for r in records])
        y = np.array([r.wear_level for r in records])
        model = regressor.get_model()

        # Línea de regresión
        X_line = np.linspace(50, 500, 100).reshape(-1, 1)
        y_line = model.predict(X_line)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color="green", label="Datos reales")
        plt.plot(X_line, y_line, color="red", label="Regresión lineal", linewidth=2)
        plt.axvline(x=hours_to_predict, color="gray", linestyle="--", label="Vehículo nuevo")
        plt.title("Predicción del Desgaste de Vehículos Militares")
        plt.xlabel("Horas de uso")
        plt.ylabel("Nivel de desgaste (%)")
        plt.legend()
        plt.grid(True)
        plt.show()


# 🧪 Ejemplo de ejecución
if __name__ == "__main__":
    example = VehicleWearPredictionExample()
    example.run()
