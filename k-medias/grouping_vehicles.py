
"""
🧠 ¿Qué es el clustering?

Clustering (o agrupamiento) es una técnica de machine learning no supervisado que sirve para encontrar grupos o patrones en los datos.
“No supervisado” significa que no necesitas etiquetas ni categorías predefinidas.
El algoritmo mira solo los datos y decide cómo agruparlos según similitudes.

🎯 Ejemplo simple:

Imagina que tienes vehículos con estas características:

Vehículo	        km/día	Eficiencia	Velocidad	Agresividad
Toyota Corolla	        40	        18	        55	0.2
Ford Mustang	        25	        10	        80	0.9
Tesla Model 3	        70	        100	        75	0.1

K-Means puede agruparlos en clusters:

Cluster 0: deportivos y agresivos → Ford Mustang, Alfa Romeo Giulia
Cluster 1: ecológicos y tranquilos → Tesla Model 3, Prius
Cluster 2: promedio, uso diario → Toyota Corolla, Honda Civic

Cada grupo tiene un centroide, que es como un “representante” del cluster.

🤔 ¿Para qué sirve?

El clustering se usa para tomar decisiones basadas en patrones reales. Algunos ejemplos:

Segmentación de clientes
Identificar clientes premium, medios o ahorradores.
Enviar promociones personalizadas.
Análisis de vehículos o flotas
Agrupar coches por estilo de conducción.
Optimizar seguros, mantenimiento o rutas.
Medicina y salud
Agrupar pacientes según síntomas, hábitos o resultados de análisis.
Marketing y productos
Clasificar productos similares para recomendación.
Detección de anomalías
Encontrar datos que no encajan en ningún grupo (outliers).    
"""

##############################################################
#  VEHICLE ANALYTICS – CLUSTERING DE VEHÍCULOS
##############################################################

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

##############################################################
# 1. CLASE VEHICLE
# Representa a un vehículo con datos de uso real.
##############################################################
class Vehicle:
    def __init__(self, model, km_per_day, fuel_efficiency, avg_speed, aggressive_driving):
        self.model = model
        self.km_per_day = km_per_day               # kilometros recorridos por día
        self.fuel_efficiency = fuel_efficiency     # km/l o eficiencia energética
        self.avg_speed = avg_speed                 # velocidad promedio
        self.aggressive_driving = aggressive_driving  # de 0 a 1

    def to_vector(self):
        return [
            self.km_per_day,
            self.fuel_efficiency,
            self.avg_speed,
            self.aggressive_driving
        ]

class VehicleClusterer:
    def __init__(self):
        self.model = None

    def fit(self, vehicles, n_clusters):
        data = [v.to_vector() for v in vehicles]
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(data)

    def predict(self, vehicle):
        return int(self.model.predict([vehicle.to_vector()])[0])

    def print_clusters(self, vehicles):
        labels = self.model.labels_
        clusters = {}

        for veh, label in zip(vehicles, labels):
            clusters.setdefault(label, []).append(veh.model)

        for cid, models in clusters.items():
            print(f"Cluster {cid}:")
            for m in models:
                print(f"  - {m}")
            print()

    def plot(self, vehicles):
        labels = self.model.labels_
        x = [v.km_per_day for v in vehicles]
        y = [v.fuel_efficiency for v in vehicles]

        plt.scatter(x, y, c=labels, s=120, cmap="viridis")
        for i, v in enumerate(vehicles):
            plt.text(x[i] + 0.2, y[i] + 0.2, v.model)
        plt.xlabel("Km por día")
        plt.ylabel("Eficiencia (km/l)")
        plt.title("Clustering de Vehículos")
        plt.grid(True)
        plt.show()


##############################################################
# 3. VEHICLE ANALYTICS – ORQUESTA ToDo
##############################################################
class VehicleAnalytics:
    def __init__(self):
        data = [
            ("Toyota Corolla", 40, 18, 55, 0.2),
            ("Ford Mustang", 25, 10, 80, 0.9),
            ("Honda Civic", 50, 20, 60, 0.3),
            ("Tesla Model 3", 70, 100, 75, 0.1),
            ("Nissan Versa", 30, 17, 50, 0.25),
            ("BMW M4", 22, 9, 85, 0.95),
            ("Audi A4", 35, 16, 70, 0.4),
            ("Kia Soul", 45, 19, 58, 0.18),
             ("Hyundai Elantra", 38, 17, 57, 0.22),
            ("Chevrolet Camaro", 28, 11, 82, 0.88),
            ("Volkswagen Golf", 48, 21, 62, 0.32),
            ("Porsche 911", 20, 8, 90, 0.97),
            ("Mazda 3", 42, 18, 59, 0.27),
            ("Mercedes C-Class", 33, 15, 72, 0.42),
            ("Subaru Impreza", 46, 19, 61, 0.2),
            ("Jaguar F-Type", 21, 7, 88, 0.93),
            ("Honda Accord", 50, 22, 63, 0.28),
            ("Ford Fiesta", 35, 14, 56, 0.25),
            ("BMW X5", 32, 13, 70, 0.4),
            ("Audi Q5", 37, 16, 68, 0.35),
            ("Chevrolet Bolt", 60, 50, 72, 0.15),
            ("Nissan Leaf", 55, 45, 68, 0.12),
            ("Tesla Model X", 75, 110, 78, 0.08),
            ("Lexus RX", 34, 15, 71, 0.38),
            ("Toyota Prius", 52, 25, 65, 0.18),
            ("Ford Explorer", 30, 14, 67, 0.3),
            ("Mini Cooper", 40, 17, 58, 0.2),
            ("Alfa Romeo Giulia", 24, 10, 83, 0.9)
        ]

        self.vehicles = [Vehicle(*row) for row in data]
        self.clusterer = VehicleClusterer()

    def run(self):
        self.clusterer.fit(self.vehicles, 3)
        self.clusterer.print_clusters(self.vehicles)

        # Predecir un vehículo nuevo
        new_car = Vehicle("Renault Zoe", 55, 22, 62, 0.15)
        print(f"El vehículo {new_car.model} pertenece al cluster:",
              self.clusterer.predict(new_car))

        self.clusterer.plot(self.vehicles)


# EJECUTAR
VehicleAnalytics().run()



