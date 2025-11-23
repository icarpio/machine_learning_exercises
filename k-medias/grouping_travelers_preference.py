"""
Agrupar viajeros según sus preferencias
🧠 Contexto

Imagina que trabajas en una agencia de viajes internacional que recibe cientos de perfiles de clientes.

Cada viajero indica cuánto le gustan distintos tipos de destinos:

🏖️ Playa
🏔️ Montaña
🏙️ Ciudad
🌄 Campo

Tu misión es desarrollar un sistema que agrupe automáticamente a los viajeros en tres grandes tipos según sus gustos.

Para lograrlo, utilizarás el algoritmo de K-Means Clustering de scikit-learn.

🎯 Objetivo del ejercicio

Debes implementar cuatro clases principales para estructurar tu solución:

1.  Traveler (almacena las preferencias de un viajero)

Atributos:

beach (int): preferencia por la playa (0–10)
mountain (int): preferencia por la montaña (0–10)
city (int): preferencia por la ciudad (0–10)
countryside (int): preferencia por el campo (0–10)

Método:

to_vector(self) -> list: devuelve las preferencias del viajero como una lista [beach, mountain, city, countryside].

2. TravelerGenerator (genera viajeros aleatorios)

Atributos:

num_travelers (int): cantidad de viajeros a generar.

Método:

generate(self) -> list[Traveler]: genera una lista de objetos Traveler con preferencias aleatorias.
Para cada preferencia, usa:

np.random.randint(0, 11)  # genera valores enteros entre 0 y 10 (inclusive)

3. TravelerClusterer (agrupa a los viajeros con K-Means)

Atributos:

model: instancia de KMeans con n_clusters=3 y random_state=42.

Métodos:

fit(self, travelers: list[Traveler]): entrena el modelo de KMeans con los vectores de preferencias.
predict(self, traveler: Traveler) -> int: devuelve el número de clúster (0, 1 o 2) al que pertenece un nuevo viajero.
get_cluster_centers(self) -> np.ndarray: retorna los centros de los clústeres calculados por el modelo.



4. TravelerClusteringExample (orquesta todo el flujo del ejemplo)

Método:

run(self): debe hacer lo siguiente:

Generar 200 viajeros usando TravelerGenerator.

Entrenar un modelo TravelerClusterer.

Mostrar en pantalla los centros de los 3 clústeres, indicando el promedio de preferencias en cada uno.

Crear un nuevo viajero personalizado, por ejemplo:

new_traveler = Traveler(beach=9, mountain=2, city=8, countryside=1)

Predecir a qué clúster pertenece ese viajero con predict.

Mostrar en pantalla los resultados.

    
"""
import numpy as np
from sklearn.cluster import KMeans

# 1. Traveler
class Traveler:
    def __init__(self, beach=0, mountain=0, city=0, countryside=0, *args, **kwargs):
        self.beach = beach
        self.mountain = mountain
        self.city = city
        self.countryside = countryside

    def to_vector(self):
        return [self.beach, self.mountain, self.city, self.countryside]

# 2. TravelerGenerator
class TravelerGenerator:
    def __init__(self, num_travelers=0, *args, **kwargs):
        self.num_travelers = num_travelers

    def generate(self):
        travelers = []
        for _ in range(self.num_travelers):
            travelers.append(
                Traveler(
                    beach=np.random.randint(0, 11),
                    mountain=np.random.randint(0, 11),
                    city=np.random.randint(0, 11),
                    countryside=np.random.randint(0, 11)
                )
            )
        return travelers

# 3. TravelerClusterer
class TravelerClusterer:
    def __init__(self, *args, **kwargs):
        self.model = KMeans(n_clusters=3, random_state=42)

    def fit(self, travelers):
        data = [t.to_vector() for t in travelers]
        self.model.fit(data)

    def predict(self, traveler):
        return int(self.model.predict([traveler.to_vector()])[0])

    def get_cluster_centers(self):
        return self.model.cluster_centers_

# 4. TravelerClusteringExample
class TravelerClusteringExample:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        generator = TravelerGenerator(200)
        travelers = generator.generate()

        clusterer = TravelerClusterer()
        clusterer.fit(travelers)

        centers = clusterer.get_cluster_centers()
        print("\n🏝️🏔️🏙️🌄  CLUSTER CENTERS:\n")
        for i, c in enumerate(centers):
            print(f"Cluster {i}: Playa={c[0]:.2f}, Montaña={c[1]:.2f}, Ciudad={c[2]:.2f}, Campo={c[3]:.2f}")

        print("\n📘 Interpretación sugerida:")
        print("- Playa + Ciudad → Viajero urbano costero")
        print("- Montaña + Campo → Amante de la naturaleza")
        print("- Valores equilibrados → Viajero versátil")

        new_traveler = Traveler(9, 2, 8, 1)
        print("\n🔍 Nuevo viajero: 9,2,8,1")
        print("📌 Cluster:", clusterer.predict(new_traveler))

# Ejecutar ejemplo
example = TravelerClusteringExample()
example.run()
