"""
Predicción para acertar la lotería



Imagina que eres parte del equipo de análisis predictivo de una empresa de juegos de azar.
Tu tarea es desarrollar un sistema capaz de analizar combinaciones de lotería y estimar cuáles tienen más probabilidad de éxito, utilizando inteligencia artificial.
Para ello, crearás un modelo de clasificación con RandomForestClassifier que aprenda de datos históricos simulados,
y luego lo aplicarás a nuevas combinaciones para seleccionar las mejores.


📌 Objetivos: 

Desarrollar una aplicación completa compuesta por varias clases, que sea capaz de:

Simular combinaciones de lotería (con 6 números únicos entre 1 y 49).
Generar datos de entrenamiento con etiquetas de "éxito" (1) o "fracaso" (0), simulando que el 10% de las combinaciones históricas fueron ganadoras.
Entrenar un modelo de machine learning para predecir la probabilidad de éxito.
Aplicar el modelo a nuevas combinaciones aleatorias.
Mostrar la mejor combinación según el modelo.
Visualizar en una gráfica las 10 combinaciones con mayor probabilidad de éxito.

🧱 Clases que debes implementar

1.  Clase GeneradorSeries

Responsable de generar combinaciones aleatorias:

Método generar_series(cantidad): genera cantidad combinaciones de 6 números únicos aleatorios entre 1 y 49.

Cada combinación debe estar ordenada para mayor coherencia visual.

2. Clase DatosLoteria

Encargada de crear los datos de entrenamiento:

Método generar_datos_entrenamiento(cantidad=1000): genera un DataFrame con 1000 combinaciones y 
una columna adicional llamada "Exito" con valores 1 (éxito) o 0 (fracaso), simulando que el 10% de las combinaciones fueron ganadoras.


3. Clase ModeloLoteria

Encargada de construir y entrenar el modelo predictivo:

Entrena un modelo de RandomForestClassifier.

Métodos:

entrenar(X, y): entrena el modelo con los datos escalados.

predecir_probabilidades(X): devuelve las probabilidades de éxito para cada combinación evaluada.

4. Clase VisualizadorResultados

Responsable de mostrar los resultados en un gráfico:

Método graficar_top_combinaciones(df_series, probabilidades, top_n=10):
Muestra un gráfico de barras horizontal con las 10 combinaciones más prometedoras.
Usa matplotlib.pyplot.


5. Clase EjecutarSimulacion

Clase principal que ejecuta todo el flujo del proyecto:

Método ejecutar():

Genera los datos simulados.
Entrena el modelo.
Genera nuevas combinaciones a evaluar.
Predice la probabilidad de éxito.
Muestra por pantalla la mejor combinación encontrada.
Muestra un gráfico con las 10 más prometedoras.



📌 Requisitos técnicos

Utiliza pandas y numpy para manipular datos.
Usa RandomForestClassifier de sklearn.ensemble.
Escala las características numéricas con StandardScaler.
Crea gráficos con matplotlib.pyplot.

🎯 Ejemplo de uso

simulacion = EjecutarSimulacion()
simulacion.ejecutar()

Salida esperada

🎯 Mejor serie encontrada:
Números: [7, 35, 39, 41, 43, 47]
Probabilidad estimada de éxito: 0.4300   
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ==============================================================
# 1. Clase GeneradorSeries
# ==============================================================
class GeneradorSeries:
    """
    Genera combinaciones aleatorias de lotería.
    Cada combinación tiene 6 números únicos entre 1 y 49.
    """

    def generar_series(self, cantidad: int) -> np.ndarray:
        series = []

        for _ in range(cantidad):
            combinacion = np.random.choice(range(1, 50), size=6, replace=False)
            combinacion.sort()  # ordenamos para coherencia visual
            series.append(combinacion)

        return np.array(series)


# ==============================================================
# 2. Clase DatosLoteria
# ==============================================================
class DatosLoteria:
    """
    Genera datos históricos simulados con etiquetas de 'Éxito' (1) o 'Fracaso' (0).
    El 10% de las combinaciones tendrán éxito.
    """

    def __init__(self):
        self.generador = GeneradorSeries()

    def generar_datos_entrenamiento(self, cantidad=1000) -> pd.DataFrame:
        series = self.generador.generar_series(cantidad)

        # etiquetamos el 10% como ganadoras
        etiquetas = np.zeros(cantidad)
        indices_ganadores = np.random.choice(cantidad, size=int(cantidad * 0.1), replace=False)
        etiquetas[indices_ganadores] = 1

        df = pd.DataFrame(series, columns=[f"N{i+1}" for i in range(6)])
        df["Exito"] = etiquetas

        return df

# ==============================================================
# 3. Clase ModeloLoteria
# ==============================================================
class ModeloLoteria:
    """
    Entrena un modelo RandomForestClassifier para estimar probabilidades de éxito.
    """

    def __init__(self):
        self.modelo = RandomForestClassifier(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.entrenado = False

    def entrenar(self, X, y):
        # Escalado
        X_scaled = self.scaler.fit_transform(X)

        # Entrenamiento
        self.modelo.fit(X_scaled, y)
        self.entrenado = True

    def predecir_probabilidades(self, X):
        if not self.entrenado:
            raise Exception("El modelo no ha sido entrenado.")

        X_scaled = self.scaler.transform(X)
        probabilidades = self.modelo.predict_proba(X_scaled)[:, 1]  # prob. de éxito (clase 1)
        return probabilidades


# ==============================================================
# 4. Clase VisualizadorResultados
# ==============================================================
class VisualizadorResultados:
    """
    Muestra un gráfico con las combinaciones más prometedoras.
    """

    def graficar_top_combinaciones(self, df_series, probabilidades, top_n=10):
        # Ordenamos de mayor a menor probabilidad
        top_indices = np.argsort(probabilidades)[-top_n:][::-1]

        top_series = df_series.iloc[top_indices]
        top_probs = probabilidades[top_indices]

        etiquetas = top_series.apply(lambda fila: "-".join(fila.astype(str)), axis=1)

        plt.figure(figsize=(10, 7))
        plt.barh(etiquetas, top_probs, color="purple")
        plt.xlabel("Probabilidad estimada de éxito")
        plt.title(f"Top {top_n} combinaciones más prometedoras")
        plt.gca().invert_yaxis()  # para mostrar la mejor arriba
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.show()


# ==============================================================
# 5. Clase EjecutarSimulacion
# ==============================================================
class EjecutarSimulacion:
    """
    Clase principal que ejecuta todo el flujo:
    - Simulación de datos
    - Entrenamiento
    - Generación de nuevas combinaciones
    - Predicción
    - Visualización
    """

    def ejecutar(self):
        print("🔄 Generando datos históricos simulados...")
        datos_loteria = DatosLoteria()
        df_entrenamiento = datos_loteria.generar_datos_entrenamiento()

        X = df_entrenamiento[[f"N{i+1}" for i in range(6)]]
        y = df_entrenamiento["Exito"]

        print("🤖 Entrenando modelo...")
        modelo = ModeloLoteria()
        modelo.entrenar(X, y)

        # generar nuevas series
        print("🎲 Generando nuevas combinaciones para evaluar...")
        generador = GeneradorSeries()
        nuevas_series = generador.generar_series(100)

        df_nuevas = pd.DataFrame(nuevas_series, columns=[f"N{i+1}" for i in range(6)])

        # predecir probabilidades
        probabilidades = modelo.predecir_probabilidades(df_nuevas.values)

        # elegir la mejor
        idx_max = np.argmax(probabilidades)
        mejor_serie = nuevas_series[idx_max]
        mejor_prob = probabilidades[idx_max]

        print("\n🎯 Mejor serie encontrada:")
        print("Números:", list(mejor_serie))
        print(f"Probabilidad estimada de éxito: {mejor_prob:.4f}")

        # Visualización
        visualizador = VisualizadorResultados()
        visualizador.graficar_top_combinaciones(df_nuevas, probabilidades)

# ==============================================================
# EJECUCIÓN DIRECTA
# ==============================================================
if __name__ == "__main__":
    simulacion = EjecutarSimulacion()
    simulacion.ejecutar()
