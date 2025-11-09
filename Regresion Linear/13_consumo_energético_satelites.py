# ======================================
# 🛰️ PREDICCIÓN DE CONSUMO ENERGÉTICO DE SATÉLITES
# ======================================

# Librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ======================================
# 1️⃣ Clase Satellite
# ======================================

class Satellite:
    """
    Representa un satélite individual con sus características básicas.
    """

    def __init__(self, duracion_mision_dias, paneles_sol, carga_util, consumo_diario):
        self.duracion_mision_dias = duracion_mision_dias     # Duración de misión (días)
        self.paneles_sol = paneles_sol              # Superficie de paneles solares (m²)
        self.carga_util = carga_util                # Carga útil (kg)
        self.consumo_diario = consumo_diario        # Energía consumida por día (kWh)

    def to_dict(self):
        """
        Convierte el satélite a un diccionario (útil para crear un DataFrame).
        """
        return {
            "duracion_mision_dias": self.duracion_mision_dias,
            "paneles_sol": self.paneles_sol,
            "carga_util": self.carga_util,
            "consumo_diario": self.consumo_diario
        }


# ======================================
# 2️⃣ Clase SatelliteDatasetGenerator
# ======================================

class SatelliteDatasetGenerator:
    """
    Genera una muestra de satélites simulados con datos aleatorios realistas.
    """

    def __init__(self, n=300):
        self.n = n  # Número de satélites a generar

    def generate(self):
        """
        Genera n satélites aleatorios con base en las fórmulas indicadas.
        """
        satellites = []

        for _ in range(self.n):
            duracion_mision_dias = np.random.uniform(100, 1000)      # días
            paneles_sol = np.random.uniform(10, 100)            # m²
            carga_util = np.random.uniform(200, 2000)           # kg

            # Fórmula del consumo diario con ruido
            consumo_diario = 5 + 0.01 * duracion_mision_dias + 0.002 * carga_util + np.random.normal(0, 1)

            # Crear objeto Satellite
            sat = Satellite(duracion_mision_dias, paneles_sol, carga_util, consumo_diario)
            satellites.append(sat)

        return satellites


# ======================================
# 3️⃣ Clase SatelliteDataProcessor
# ======================================

class SatelliteDataProcessor:
    """
    Convierte la lista de satélites a un DataFrame y calcula la eficiencia energética.
    """

    def __init__(self, satellites):
        # Convertimos los objetos Satellite en un DataFrame
        self.df = pd.DataFrame([s.to_dict() for s in satellites])

        # Nueva métrica: eficiencia energética (kWh/m²)
        self.df["eficiencia_energia"] = self.df["consumo_diario"] / self.df["paneles_sol"]

    def get_dataframe(self):
        """
        Devuelve el DataFrame procesado.
        """
        return self.df


# ======================================
# 4️⃣ Clase EnergyConsumptionRegressor
# ======================================

class EnergyConsumptionRegressor:
    """
    Modelo de regresión lineal para predecir el consumo diario a partir de la duración de la misión.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Entrena el modelo de regresión lineal y devuelve las predicciones.
        """
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate(self, y_true, y_pred):
        """
        Calcula el coeficiente de determinación R².
        """
        return r2_score(y_true, y_pred)

    def get_coefficients(self):
        """
        Devuelve la pendiente (coef) y el intercepto (b) de la recta de regresión.
        """
        return self.model.coef_[0], self.model.intercept_


# ======================================
# 5️⃣ Clase SatellitePlotter
# ======================================

class SatellitePlotter:
    """
    Genera un gráfico del consumo diario vs. duración de la misión, coloreado por carga útil.
    """

    def __init__(self, df, y_pred):
        self.df = df
        self.y_pred = y_pred

    def plot(self):
        """
        Crea un gráfico de dispersión con línea de regresión.
        """
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.df["duracion_mision_dias"],
            self.df["consumo_diario"],
            c=self.df["carga_util"],
            cmap="viridis",
            alpha=0.7,
            label="Datos reales"
        )

        # Línea de regresión en rojo
        plt.plot(self.df["duracion_mision_dias"], self.y_pred, color="red", linewidth=2, label="Regresión lineal")

        # Etiquetas y colorbar
        plt.colorbar(scatter, label="Carga útil (kg)")
        plt.title("Consumo energético diario vs Duración de misión")
        plt.xlabel("Duración de misión (días)")
        plt.ylabel("Consumo diario (kWh)")
        plt.legend()
        plt.grid(True)
        plt.show()


# ======================================
# 6️⃣ Clase SatelliteAnalysisExample (orquestador)
# ======================================

class SatelliteAnalysisExample:
    """
    Ejecuta todo el flujo completo: generar, procesar, modelar y visualizar.
    """

    def run(self):
        print("🚀 Generando datos sintéticos...")
        generator = SatelliteDatasetGenerator(n=300)
        satellites = generator.generate()

        print("🔍 Procesando datos...")
        processor = SatelliteDataProcessor(satellites)
        df = processor.get_dataframe()

        print("📈 Aplicando regresión lineal...")
        regressor = EnergyConsumptionRegressor()

        X = df[["duracion_mision_dias"]]  # Variable predictora
        y = df["consumo_diario"]     # Variable dependiente

        y_pred = regressor.fit(X, y)
        r2 = regressor.evaluate(y, y_pred)
        coef, intercept = regressor.get_coefficients()

        print(f"\n📊 RESULTADOS DEL MODELO:")
        print(f"Coeficiente (pendiente): {coef:.4f}")
        print(f"Intercepto: {intercept:.4f}")
        print(f"R² del modelo: {r2:.4f}")

        print("\n🎨 Generando visualización...")
        plotter = SatellitePlotter(df, y_pred)
        plotter.plot()


# ======================================
# ✅ EJEMPLO DE USO
# ======================================

if __name__ == "__main__":
    example = SatelliteAnalysisExample()
    example.run()
