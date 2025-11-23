import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. CLASE: SimuladorViviendas
# ============================================================
class SimuladorViviendas:
    """
    Genera un conjunto de datos sintético con información de viviendas.
    """

    def __init__(self, n=200, seed=42):
        """
        Constructor
        :param n: número de registros
        :param seed: semilla para reproducibilidad
        """
        self.n = n
        self.seed = seed

    def generar_datos(self) -> pd.DataFrame:
        """
        Genera un DataFrame con las características de viviendas y su precio.
        :return: DataFrame con columnas:
            - Superficie (m2)
            - Habitaciones
            - Antigüedad (años)
            - Distancia_centro (km)
            - Baños
            - Precio (valor objetivo)
        """

        np.random.seed(self.seed)

        superficie = np.random.uniform(50, 200, self.n)
        habitaciones = np.random.randint(1, 6, self.n)
        antigüedad = np.random.randint(0, 50, self.n)
        distancia = np.random.uniform(0.5, 20, self.n)
        baños = np.random.randint(1, 4, self.n)

        # Fórmula aproximada con ruido para generar el precio realista
        precio = (
            superficie * 2500 +
            habitaciones * 15000 +
            baños * 10000 -
            antigüedad * 1200 -
            distancia * 5000 +
            np.random.normal(0, 30000, self.n)  # ruido aleatorio
        )

        df = pd.DataFrame({
            "Superficie": superficie,
            "Habitaciones": habitaciones,
            "Antigüedad": antigüedad,
            "Distancia_centro": distancia,
            "Baños": baños,
            "Precio": precio
        })

        return df


# ============================================================
# 2. CLASE: ModeloPrecioVivienda
# ============================================================
class ModeloPrecioVivienda:
    """
    Representa un modelo de regresión lineal para predecir precios de viviendas.
    """

    def __init__(self):
        # Creamos el modelo de regresión lineal
        self.modelo = LinearRegression()

    def entrenar(self, data: pd.DataFrame):
        """
        Entrena el modelo usando los datos proporcionados.
        """

        # Variables independientes
        X = data.drop("Precio", axis=1)

        # Variable objetivo
        y = data["Precio"]

        # División de los datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Entrenamiento
        self.modelo.fit(self.X_train, self.y_train)

        print("Modelo entrenado correctamente.\n")

    def evaluar(self):
        """
        Evalúa el modelo mostrando MSE y R2.
        """

        pred = self.modelo.predict(self.X_test)

        mse = mean_squared_error(self.y_test, pred)
        r2 = r2_score(self.y_test, pred)

        print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
        print(f"R² del modelo: {r2:.2f}\n")

    def predecir(self, nueva_vivienda: pd.DataFrame) -> float:
        """
        Recibe un DataFrame con una vivienda y devuelve el precio estimado.
        """

        pred = self.modelo.predict(nueva_vivienda)[0]
        return pred


# ============================================================
# 3. CLASE: TestModeloPrecio
# ============================================================
class TestModeloPrecio:
    """
    Clase que integra la simulación, entrenamiento, evaluación y predicción.
    """

    def ejecutar(self):
        # 1. Generar datos
        sim = SimuladorViviendas()
        datos = sim.generar_datos()

        print("Primeras filas de datos simulados:")
        print(datos.head(), "\n")

        # 2. Entrenar modelo
        modelo = ModeloPrecioVivienda()
        modelo.entrenar(datos)

        # 3. Evaluar modelo
        modelo.evaluar()

        # 4. Crear vivienda de ejemplo
        nueva = pd.DataFrame({
            "Superficie": [120],
            "Habitaciones": [3],
            "Antigüedad": [10],
            "Distancia_centro": [5],
            "Baños": [2]
        })

        # 5. Predicción
        precio_estimado = modelo.predecir(nueva)

        print(f"El precio estimado de la vivienda es: ${precio_estimado:,.2f}")


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    test = TestModeloPrecio()
    test.ejecutar()




"""
🎉 ¿Qué incluye este código?

✔ SimuladorViviendas

Genera aleatoriamente:

Superficie (50–200 m²)
Habitaciones (1–5)
Antigüedad (0–50 años)
Distancia al centro (0.5–20 km)
Baños (1–3)
Precio (calculado con fórmula + ruido)

✔ ModeloPrecioVivienda

Entrena regresión lineal
Calcula MSE y R²
Predice el precio de nuevas viviendas

✔ TestModeloPrecio

Ejecuta todo junto
Imprime los primeros datos
Entrena y evalúa
Predice una vivienda de ejemplo
"""