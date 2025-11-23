"""

💤 Análisis de Perfiles de Sueño con K-Means

Una empresa especializada en salud del sueño desea analizar los hábitos de descanso de cientos de personas.
Se han recogido datos sobre cómo duermen los usuarios, y ahora tu misión es agrupar a los individuos en diferentes tipos de "dormidores" usando aprendizaje no supervisado.


🎯 Objetivo

Crear un sistema en Python que genere perfiles de sueño sintéticos, los agrupe en 3 categorías usando K-Means Clustering, y visualice los resultados.


🧩 Estructura que debes seguir

1. SleepProfile

Clase que representa el perfil de sueño de una persona.

Atributos:

duration: duración promedio del sueño en horas (ej. 7.5)
latency: tiempo promedio para quedarse dormido en minutos (ej. 25)
wakeups: cantidad de despertares durante la noche (ej. 2)
variability: variabilidad en la hora de irse a dormir, en minutos (ej. 35)

Métodos:

to_vector(self): Devuelve la información del perfil como una lista [duration, latency, wakeups, variability].

2. SleepDatasetGenerator

Clase que genera una lista de objetos SleepProfile con datos sintéticos y realistas.

Constructor:

__init__(self, n=300): genera n perfiles (por defecto 300).

Método:

generate(self): retorna una lista de objetos SleepProfile, donde los atributos se generan así:
duration: distribución normal con media 7 horas y desviación estándar 1.2 → np.random.normal(7, 1.2, n)
latency: distribución normal positiva (usar np.abs) con media 20 y std 10
wakeups: distribución Poisson con media 1.5 → np.random.poisson(1.5, n)
variability: distribución normal positiva con media 30 y std 15

3. SleepClusterer

Clase que entrena un modelo K-Means y agrupa los perfiles.

Constructor:

__init__(self, n_clusters=3): crea un modelo KMeans con 3 clústeres y un StandardScaler para escalar los datos antes de entrenar.

Métodos:
fit(self, profiles): recibe una lista de SleepProfile, los convierte a vectores, normaliza con StandardScaler y entrena el modelo. Retorna X_scaled y las etiquetas de clúster (labels).
get_cluster_centers(self): devuelve los centros de clúster transformados a su escala original (usa scaler.inverse_transform).



4. SleepAnalysisExample

Clase que ejecuta todo el análisis.

Método:

run(self):

Genera los perfiles usando SleepDatasetGenerator.

Agrupa los perfiles usando SleepClusterer con 3 grupos.
Crea un DataFrame para mostrar resultados.
Imprime los centroides (promedios de cada grupo).
Grafica los perfiles usando matplotlib (eje X: duración, eje Y: variabilidad).

🛠 Requisitos mínimos

Usar numpy para generar datos sintéticos.
Usar scikit-learn para KMeans y StandardScaler.
Usar matplotlib para graficar los resultados.
Organizar el código en clases separadas como se indica.
Mostrar en consola los centroides de cada grupo.
Graficar los grupos usando diferentes colores.



📈 ¿Qué deberías mostrar al final?

Los centroides de los grupos, interpretando lo que caracteriza a cada uno.
Un gráfico de dispersión donde se vea la agrupación de perfiles por:

Eje X: Duración del sueño
Eje Y: Variabilidad de hora de dormir

Comentarios sobre posibles tipos de durmientes: ¿hay un grupo de "insomnes"? ¿otro de "buenos durmientes"?

🧪 Ejemplo de uso

example = SleepAnalysisExample()
example.run()

Salida esperada

📌 Centroides de los grupos:
Grupo 0: Duración=6.30h, Latencia=19.3min, Despertares=1.2, Variabilidad=39.6min
Grupo 1: Duración=6.79h, Latencia=18.9min, Despertares=3.4, Variabilidad=26.5min
Grupo 2: Duración=7.98h, Latencia=18.6min, Despertares=1.0, Variabilidad=22.3min
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

##############################################################
# 1. SleepProfile
##############################################################
class SleepProfile:
    def __init__(self, duration, latency, wakeups, variability):
        self.duration = duration
        self.latency = latency
        self.wakeups = wakeups
        self.variability = variability

    def to_vector(self):
        """Devuelve los atributos como lista para K-Means"""
        return [self.duration, self.latency, self.wakeups, self.variability]


##############################################################
# 2. SleepDatasetGenerator
##############################################################
class SleepDatasetGenerator:
    def __init__(self, n=300):
        self.n = n

    def generate(self):
        durations = np.random.normal(7, 1.2, self.n)
        latencies = np.abs(np.random.normal(20, 10, self.n))
        wakeups = np.random.poisson(1.5, self.n)
        variability = np.abs(np.random.normal(30, 15, self.n))

        profiles = []
        for d, l, w, v in zip(durations, latencies, wakeups, variability):
            profiles.append(SleepProfile(d, l, w, v))
        return profiles


##############################################################
# 3. SleepClusterer
##############################################################
class SleepClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # atributo correcto
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.labels = None

    def fit(self, profiles):
        X = np.array([p.to_vector() for p in profiles])
        self.X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(self.X_scaled)
        self.labels = self.kmeans.labels_
        return self.X_scaled, self.labels

    def get_cluster_centers(self):
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)
        return centers_original

##############################################################
# 4. SleepAnalysisExample
##############################################################
class SleepAnalysisExample:
    def __init__(self):
        pass

    def run(self):
        # 1️⃣ Generar datos sintéticos
        generator = SleepDatasetGenerator(n=300)
        profiles = generator.generate()

        # 2️⃣ Entrenar clusterer
        clusterer = SleepClusterer(n_clusters=3)
        _, labels = clusterer.fit(profiles)

        # 3️⃣ Crear DataFrame para mostrar resultados
        df = pd.DataFrame([p.to_vector() for p in profiles],
                          columns=['Duration', 'Latency', 'Wakeups', 'Variability'])
        df['Cluster'] = labels

        # 4️⃣ Mostrar centroides
        centers = clusterer.get_cluster_centers()
        print("\n📌 Centroides de los grupos:")
        for i, c in enumerate(centers):
            print(f"Grupo {i}: Duración={c[0]:.2f}h, Latencia={c[1]:.1f}min, "
                  f"Despertares={c[2]:.1f}, Variabilidad={c[3]:.1f}min")

        # 5️⃣ Gráfico de dispersión
        plt.figure(figsize=(8,6))
        colors = ['red', 'green', 'blue']
        for i in range(clusterer.n_clusters):
            cluster_points = df[df['Cluster'] == i]
            plt.scatter(cluster_points['Duration'], cluster_points['Variability'],
                        c=colors[i], label=f'Grupo {i}', alpha=0.6)
        plt.scatter(centers[:,0], centers[:,3], c='black', marker='X', s=150, label='Centroides')
        plt.xlabel('Duración del sueño (h)')
        plt.ylabel('Variabilidad de hora de dormir (min)')
        plt.title('Clustering de perfiles de sueño')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 6️⃣ Interpretación aproximada
        print("\n💡 Posibles tipos de durmientes según clusters:")
        for i, c in enumerate(centers):
            if c[0] < 6.5:
                tipo = "Posible insomne / duerme poco"
            elif c[0] > 7.5:
                tipo = "Buen durmiente / duerme suficiente"
            else:
                tipo = "Durmiente promedio / regular"
            print(f"Grupo {i}: {tipo}")

# Ejecutar ejemplo
example = SleepAnalysisExample()
example.run()
