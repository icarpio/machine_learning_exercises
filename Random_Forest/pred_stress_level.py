""""
Predicción del nivel de estrés

🌍 Contexto

Los niveles de estrés afectan directamente a la salud física y mental.
En este proyecto, trabajarás como si fueras parte del equipo de desarrollo de un sistema de monitoreo de estrés para deportistas de alto rendimiento o trabajadores 
en ambientes exigentes.
Se te ha encomendado diseñar un clasificador que, a partir de tres medidas fisiológicas, pueda predecir el nivel de estrés de una persona.
Para ello, deberás simular datos realistas, entrenar un modelo de aprendizaje automático y visualizar los resultados.

🎯 Objetivos

Simular datos fisiológicos (ritmo cardíaco, nivel de cortisol y conductancia de la piel).

Clasificar el nivel de estrés de las personas como:

🟢 Bajo, 🟠 Moderado o 🔴 Alto.
Entrenar un clasificador Random Forest.
Evaluar el rendimiento del modelo.
Realizar predicciones personalizadas.
Visualizar los datos y resultados con gráficos interpretables.

🛠️ Requisitos Técnicos

Debes usar:

NumPy para generar datos.
Pandas para manipular estructuras.
matplotlib.pyplot para visualizar.
sklearn para entrenamiento del modelo y métricas.
Programación orientada a objetos (clases bien definidas).

👨‍🔬 Parte 1: Clase para representar individuos

Crea una clase llamada Individual con los siguientes atributos:

Ritmo cardíaco (heart_rate) en pulsaciones por minuto.
Nivel de cortisol (cortisol_level) en µg/dL.
Conductancia de la piel (skin_conductance) en µS.
Nivel de estrés (stress_level): cadena de texto ('Bajo', 'Moderado' o 'Alto').
Incluye un método to_vector() que devuelva solo las tres primeras variables como lista.

🧪 Parte 2: Simulador de datos

Crea una clase StressDataGenerator que genere una lista de objetos Individual con valores aleatorios realistas:

Ritmo cardíaco: media 75, desviación estándar 15.
Cortisol: media 12, desviación estándar 4.
Conductancia: media 5, desviación estándar 1.5.
Clasifica los individuos según estas reglas:

🔴 Alto: si cualquiera de las tres medidas supera estos umbrales:

Ritmo cardíaco > 90
Cortisol > 18
Conductancia > 6.5

🟠 Moderado: si alguna supera:

Ritmo cardíaco > 70
Cortisol > 10
Conductancia > 4.5
pero no cumple los criterios de "Alto".

🟢 Bajo: si ninguna medida supera esos valores.

🤖 Parte 3: Clasificador con Random Forest

Crea una clase StressClassifier con los métodos:

fit(individuals) → entrena el modelo con datos.
predict(heart_rate, cortisol, conductance) → devuelve el nivel de estrés estimado.
evaluate(test_data) → imprime matriz de confusión e informe de clasificación.

🔍 Parte 4: Ejecución completa del análisis

Crea una clase llamada StressAnalysisExample que se encargue de ejecutar todo el flujo del proyecto. Esta clase debe implementar un método run() que realice las siguientes tareas:

Generación de datos:
Genera 300 individuos simulados usando la clase StressDataGenerator.

Entrenamiento y evaluación del modelo:
Divide los datos en dos subconjuntos: 70% para entrenamiento y 30% para prueba.
Entrena un clasificador usando la clase StressClassifier.

Evalúa el rendimiento del modelo mostrando:

La matriz de confusión.

El informe de clasificación con precisión, recall y f1-score.

Predicción personalizada:

Utiliza el modelo entrenado para predecir el nivel de estrés de un individuo con las siguientes características:

Ritmo cardíaco: 95
Cortisol: 20
Conductancia: 7
Muestra por pantalla la predicción realizada.

Visualización de los datos:

Convierte los datos generados en un DataFrame de pandas.
Crea un gráfico de dispersión con matplotlib:
Eje X: nivel de cortisol.
Eje Y: ritmo cardíaco.

Color de los puntos según el nivel de estrés:

🟢 Verde → Bajo
🟠 Naranja → Moderado
🔴 Rojo → Alto
Agrega título, leyenda y cuadrícula para facilitar la interpretación visual.

"""


# ===========================================
# 📦 IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ===========================================
# 🧍‍♂️ PARTE 1: CLASE INDIVIDUAL
# ===========================================
class Individual:
    """
    Representa a un individuo con sus medidas fisiológicas.
    """

    def __init__(self, heart_rate, cortisol_level, skin_conductance, stress_level):
        self.heart_rate = heart_rate
        self.cortisol_level = cortisol_level
        self.skin_conductance = skin_conductance
        self.stress_level = stress_level

    def to_vector(self):
        """
        Devuelve las variables fisiológicas como lista (para entrenar el modelo).
        """
        return [self.heart_rate, self.cortisol_level, self.skin_conductance]


# ===========================================
# 🧪 PARTE 2: GENERADOR DE DATOS
# ===========================================
class StressDataGenerator:
    """
    Genera individuos simulados con valores fisiológicos aleatorios realistas.
    """

    def __init__(self, n_individuals=300):
        self.n_individuals = n_individuals

    def generate(self):
        individuals = []

        for _ in range(self.n_individuals):
            heart_rate = np.random.normal(75, 15)
            cortisol = np.random.normal(12, 4)
            conductance = np.random.normal(5, 1.5)

            # Clasificación según umbrales definidos
            if heart_rate > 90 or cortisol > 18 or conductance > 6.5:
                stress = "Alto"
            elif heart_rate > 70 or cortisol > 10 or conductance > 4.5:
                stress = "Moderado"
            else:
                stress = "Bajo"

            individuals.append(Individual(heart_rate, cortisol, conductance, stress))

        return individuals


# ===========================================
# 🤖 PARTE 3: CLASIFICADOR
# ===========================================
class StressClassifier:
    """
    Clasificador de niveles de estrés con Random Forest.
    """

    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, individuals):
        X = [ind.to_vector() for ind in individuals]
        y = [ind.stress_level for ind in individuals]
        self.model.fit(X, y)

    def predict(self, heart_rate, cortisol, conductance):
        X_new = [[heart_rate, cortisol, conductance]]
        return self.model.predict(X_new)[0]

    def evaluate(self, test_data):
        X_test = [ind.to_vector() for ind in test_data]
        y_true = [ind.stress_level for ind in test_data]
        y_pred = self.model.predict(X_test)

        print("\n📊 Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_true, y_pred))


# ===========================================
# 🔍 PARTE 4: EJECUCIÓN COMPLETA DEL ANÁLISIS
# ===========================================
class StressAnalysisExample:
    """
    Ejecuta todo el flujo: generación, entrenamiento, evaluación, predicción y visualización.
    """

    def run(self):
        print("\n🚀 INICIANDO ANÁLISIS DE ESTRÉS...\n")

        # 1️⃣ Generar datos
        print("📈 Generando datos simulados...")
        generator = StressDataGenerator(n_individuals=300)
        data = generator.generate()

        # 2️⃣ Separar entrenamiento / prueba
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        # 3️⃣ Entrenar modelo
        print("\n🧠 Entrenando modelo Random Forest...")
        classifier = StressClassifier()
        classifier.fit(train_data)

        # 4️⃣ Evaluar modelo
        print("\n📉 Evaluando modelo con datos de prueba...")
        classifier.evaluate(test_data)

        # 5️⃣ PREDICCIÓN CON INPUTS DEL USUARIO 🎯
        print("\n👤 Vamos a predecir tu nivel de estrés personalizado.")
        print("Introduce tus valores fisiológicos (usa números decimales si hace falta):\n")

        try:
            hr = float(input("💓 Ritmo cardíaco (bpm): "))
            cort = float(input("🧪 Nivel de cortisol (µg/dL): "))
            cond = float(input("⚡ Conductancia de la piel (µS): "))
        except ValueError:
            print("\n⚠️ Error: debes introducir números válidos.")
            return

        prediction = classifier.predict(hr, cort, cond)

        print("\n🔮 RESULTADO DE LA PREDICCIÓN:")
        print(f"  Ritmo cardíaco: {hr}")
        print(f"  Cortisol: {cort}")
        print(f"  Conductancia: {cond}")
        print(f"  → Nivel estimado de estrés: 🧠 {prediction.upper()}")

        # 6️⃣ Visualización de datos
        print("\n🎨 Mostrando visualización de datos simulados...")
        df = pd.DataFrame([{
            "Ritmo cardíaco": ind.heart_rate,
            "Cortisol": ind.cortisol_level,
            "Conductancia": ind.skin_conductance,
            "Estrés": ind.stress_level
        } for ind in data])

        colores = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}

        plt.figure(figsize=(8, 6))
        for nivel, color in colores.items():
            subset = df[df["Estrés"] == nivel]
            plt.scatter(subset["Cortisol"], subset["Ritmo cardíaco"],
                        label=nivel, color=color, alpha=0.7)

        plt.title("Relación entre Cortisol y Ritmo Cardíaco según nivel de Estrés")
        plt.xlabel("Nivel de Cortisol (µg/dL)")
        plt.ylabel("Ritmo Cardíaco (bpm)")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("\n✅ Análisis completado con éxito.\n")


# ===========================================
# 🧠 EJEMPLO DE USO
# ===========================================
if __name__ == "__main__":
    example = StressAnalysisExample()
    example.run()
