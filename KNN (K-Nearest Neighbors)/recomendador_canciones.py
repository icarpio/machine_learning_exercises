"""
Recomendador de canciones inteligente
🧠 Contexto

Estás desarrollando un sistema para una plataforma musical que quiere ofrecer recomendaciones automáticas basadas en características cuantitativas de las canciones, como su energía o duración.
Utilizarás el algoritmo K-Nearest Neighbors (KNN) de la biblioteca scikit-learn para encontrar las canciones más similares a una canción objetivo.



🎯 Objetivo del ejercicio

Implementar un sistema de recomendación de canciones en Python, usando el modelo de K Vecinos Más Cercanos de scikit-learn.
El sistema debe permitir recomendar canciones similares a partir de características musicales numéricas.



📌 Requisitos

🧩 1. Clase Song

Crea una clase Song que represente una canción, con los siguientes atributos:

title (str): título de la canción.
artist (str): artista o grupo musical.
energy (float): energía de la canción (0.4 a 1.0).
danceability (float): cuán bailable es la canción (0.4 a 1.0).
duration (int): duración en segundos (180 a 300).
popularity (int): nivel de popularidad (50 a 100).

La clase debe incluir:

Un método to_vector() que devuelva una lista con los valores [energy, danceability, duration, popularity].
Un método __str__() que permita imprimir la canción en formato "Song Title by Artist".



🤖 2. Clase SongRecommender

Crea una clase SongRecommender que use el algoritmo de KNN de scikit-learn:

El constructor debe aceptar un parámetro k (número de vecinos a considerar).
El método fit(song_list) debe:
Convertir la lista de canciones en una matriz de características numéricas.
Ajustar el modelo NearestNeighbors con estas características.
El método recommend(target_song) debe:
Obtener los k vecinos más cercanos a la canción objetivo.
Devolver la lista de canciones recomendadas (sin incluir la propia canción objetivo si aparece).



🔁 3. Clase SongGenerator

Crea una clase SongGenerator con:

Un parámetro num_songs (por defecto 30).
Un método generate() que genere canciones aleatorias con numpy, usando nombres como "Song1", "Song2", etc., y artistas "Artist1", "Artist2", etc.



🧪 4. Clase SongRecommendationExample

Crea una clase de ejemplo que:

Genere una lista de canciones con SongGenerator.
Defina una canción personalizada como objetivo (target_song).
Cree una instancia de SongRecommender, la entrene con las canciones y obtenga recomendaciones.
Imprima por pantalla las canciones recomendadas.


💡 Recomendaciones para completar el ejercicio

Usa numpy para generar valores aleatorios.
Recuerda importar NearestNeighbors desde sklearn.neighbors.
Asegúrate de convertir los objetos Song a vectores antes de ajustar o predecir con el modelo.
No incluyas la canción objetivo entre las recomendaciones (verifica si es necesario).

"""


import numpy as np
from sklearn.neighbors import NearestNeighbors

# 🧩 1. Clase Song
class Song:
    def __init__(self, title, artist, energy, danceability, duration, popularity):
        self.title = title
        self.artist = artist
        self.energy = energy
        self.danceability = danceability
        self.duration = duration
        self.popularity = popularity

    def to_vector(self):
        """Devuelve las características de la canción como lista"""
        return [self.energy, self.danceability, self.duration, self.popularity]

    def __str__(self):
        return f"{self.title} by {self.artist}"


# 🤖 2. Clase SongRecommender
class SongRecommender:
    def __init__(self, k=5):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 para excluir la propia canción
        self.songs = None

    def fit(self, song_list):
        """Entrena el modelo KNN con la lista de canciones"""
        self.songs = song_list
        feature_matrix = np.array([song.to_vector() for song in song_list])
        self.model.fit(feature_matrix)

    def recommend(self, target_song):
        """Recomienda k canciones similares a target_song"""
        target_vector = np.array(target_song.to_vector()).reshape(1, -1)
        distances, indices = self.model.kneighbors(target_vector)
        recommendations = []
        for idx in indices[0]:
            recommended_song = self.songs[idx]
            if recommended_song.title != target_song.title or recommended_song.artist != target_song.artist:
                recommendations.append(recommended_song)
            if len(recommendations) == self.k:
                break
        return recommendations


# 🔁 3. Clase SongGenerator
class SongGenerator:
    def __init__(self, num_songs=30):
        self.num_songs = num_songs

    def generate(self):
        """Genera canciones aleatorias"""
        songs = []
        for i in range(1, self.num_songs + 1):
            title = f"Song{i}"
            artist = f"Artist{np.random.randint(1, 6)}"
            energy = np.round(np.random.uniform(0.4, 1.0), 2)
            danceability = np.round(np.random.uniform(0.4, 1.0), 2)
            duration = np.random.randint(180, 301)
            popularity = np.random.randint(50, 101)
            songs.append(Song(title, artist, energy, danceability, duration, popularity))
        return songs


# 🧪 4. Clase SongRecommendationExample
class SongRecommendationExample:
    def run(self):
        # Generar canciones
        generator = SongGenerator()
        song_list = generator.generate()

        # Definir canción objetivo
        target_song = Song("Mi Canción", "Mi Artista", energy=0.8, danceability=0.9, duration=240, popularity=90)

        # Entrenar el recomendador
        recommender = SongRecommender(k=3)
        recommender.fit(song_list)

        # Obtener recomendaciones
        recommendations = recommender.recommend(target_song)

        # Mostrar resultados
        print(f"🎵 Recomendaciones para '{target_song.title}':")
        for song in recommendations:
            print(f" - {song}")


# ✅ Ejecutar ejemplo
if __name__ == "__main__":
    example = SongRecommendationExample()
    example.run()



