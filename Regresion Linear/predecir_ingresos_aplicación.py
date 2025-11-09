"""
Predecir ingresos de una aplicación
📘 Enunciado del ejercicio:

Eres parte de un equipo de analistas de datos en una empresa tecnológica que desarrolla aplicaciones móviles. Te han proporcionado un pequeño conjunto de datos con información sobre diferentes apps que ya están publicadas, y tu tarea es crear un modelo de regresión lineal para predecir los ingresos estimados de una nueva app.



📊 Datos disponibles por app:

app_name: Nombre de la app
downloads: Número de descargas (en miles)
rating: Valoración media de los usuarios (de 1 a 5)
size_mb: Tamaño de la app (en MB)
reviews: Número de valoraciones escritas
revenue: Ingresos generados (en miles de dólares) → variable a predecir



📌 Tareas que debes realizar:

Crea una clase App que represente cada app con sus atributos.
Implementa un método .to_features() que devuelva una lista con las características relevantes (downloads, rating, size_mb, reviews), excluyendo revenue.

Crea una clase RevenuePredictor que:
Reciba una lista de objetos App (con revenue conocido).
Extraiga las características relevantes para entrenar un modelo.
Entrene un modelo de regresión lineal para predecir los ingresos (revenue).
Permita predecir los ingresos de una nueva app con datos similares.
Ignora las apps que tengan revenue=None al entrenar el modelo.
El método predict() debe devolver un float (por ejemplo, 207.59).
Puedes utilizar LinearRegression de sklearn.linear_model para implementar el modelo.

🧪 Ejemplo de uso

# Datos simulados de entrenamiento
training_apps = [
    App("TaskPro", 200, 4.2, 45.0, 1800, 120.0),
    App("MindSpark", 150, 4.5, 60.0, 2100, 135.0),
    App("WorkFlow", 300, 4.1, 55.0, 2500, 160.0),
    App("ZenTime", 120, 4.8, 40.0, 1700, 140.0),
    App("FocusApp", 180, 4.3, 52.0, 1900, 130.0),
    App("BoostApp", 220, 4.0, 48.0, 2300, 145.0),
]
 
# Creamos y entrenamos el predictor
predictor = RevenuePredictor()
predictor.fit(training_apps)
 
# Nueva app para predecir
new_app = App("FocusMaster", 250, 4.5, 50.0, 3000)
predicted_revenue = predictor.predict(new_app)
print(f"Ingresos estimados para {new_app.name}: ${predicted_revenue:.2f}K")


Salida esperada
Ingresos estimados para FocusMaster: $207.59K
"""


# ============================================
# 📘 PREDICCIÓN DE INGRESOS DE UNA APLICACIÓN
# ============================================

# Importamos las librerías necesarias
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------------------
# Clase App: Representa una aplicación móvil
# --------------------------------------------
class App:
    def __init__(self, name, downloads, rating, size_mb, reviews, revenue=None):
        """
        Constructor de la clase App.

        Parámetros:
        - name (str): nombre de la aplicación.
        - downloads (float): número de descargas (en miles).
        - rating (float): valoración media de los usuarios (1 a 5).
        - size_mb (float): tamaño de la app en MB.
        - reviews (int): número de valoraciones escritas.
        - revenue (float): ingresos generados (en miles de dólares).
        """
        self.name = name
        self.downloads = downloads
        self.rating = rating
        self.size_mb = size_mb
        self.reviews = reviews
        self.revenue = revenue

    def to_features(self):
        """
        Devuelve las características relevantes de la app
        para el modelo de predicción (excluye revenue).
        """
        return [self.downloads, self.rating, self.size_mb, self.reviews]


# --------------------------------------------
# Clase RevenuePredictor: Entrena y predice ingresos
# --------------------------------------------
class RevenuePredictor:
    def __init__(self):
        """
        Inicializa el predictor con un modelo de regresión lineal vacío.
        """
        self.model = LinearRegression()
        self.is_trained = False  # Indicador para saber si el modelo ya fue entrenado

    def fit(self, apps):
        """
        Entrena el modelo de regresión lineal usando una lista de objetos App.

        Ignora aquellas apps cuyo revenue sea None.
        """
        # Filtramos solo las apps con ingresos conocidos
        training_data = [app for app in apps if app.revenue is not None]

        # Si no hay datos válidos, lanzamos un error
        if not training_data:
            raise ValueError("No hay datos válidos con revenue para entrenar el modelo.")

        # Extraemos las características (X) y los ingresos (y)
        X = np.array([app.to_features() for app in training_data])
        y = np.array([app.revenue for app in training_data])

        # Entrenamos el modelo
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, app):
        """
        Predice los ingresos (en miles de dólares) de una app dada.

        Parámetros:
        - app (App): aplicación nueva sin revenue.

        Retorna:
        - float: ingresos estimados.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo aún no ha sido entrenado. Llama a fit() primero.")

        # Convertimos la app a un array de características
        X_new = np.array(app.to_features()).reshape(1, -1)

        # Realizamos la predicción
        predicted_revenue = self.model.predict(X_new)[0]

        return float(predicted_revenue)


# --------------------------------------------
# 🧪 Ejemplo de uso
# --------------------------------------------

if __name__ == "__main__":
    # Datos simulados de entrenamiento
    training_apps = [
        App("TaskPro", 200, 4.2, 45.0, 1800, 120.0),
        App("MindSpark", 150, 4.5, 60.0, 2100, 135.0),
        App("WorkFlow", 300, 4.1, 55.0, 2500, 160.0),
        App("ZenTime", 120, 4.8, 40.0, 1700, 140.0),
        App("FocusApp", 180, 4.3, 52.0, 1900, 130.0),
        App("BoostApp", 220, 4.0, 48.0, 2300, 145.0),
    ]

    # Creamos y entrenamos el predictor
    predictor = RevenuePredictor()
    predictor.fit(training_apps)

    # Nueva app para predecir
    new_app = App("FocusMaster", 250, 4.5, 50.0, 3000)

    # Realizamos la predicción
    predicted_revenue = predictor.predict(new_app)

    # Mostramos el resultado
    print(f"Ingresos estimados para {new_app.name}: ${predicted_revenue:.2f}K")
