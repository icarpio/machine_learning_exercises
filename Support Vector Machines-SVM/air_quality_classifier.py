"""
Clasificador de calidad del aire
Contexto

Trabajas en una empresa de tecnología verde que quiere monitorizar la calidad del aire para proteger la salud de las personas. 
Debes crear un modelo que, a partir de medidas de contaminantes en el aire, prediga si un área es saludable o está contaminada.

Objetivo

Construir un sistema en Python que:

-Genere datos sintéticos de calidad del aire con medidas de PM2.5, PM10, O3 y NO2.
-Entrene un clasificador SVM para distinguir entre aire saludable (0) y contaminado (1).
-Permita predecir la calidad del aire de nuevas muestras.

Requisitos técnicos

1. Clase AirSample

Representa una muestra de calidad del aire con los siguientes atributos:

pm25: concentración de partículas finas PM2.5 (µg/m³)
pm10: concentración de partículas gruesas PM10 (µg/m³)
o3: concentración de ozono (ppb)
no2: concentración de dióxido de nitrógeno (ppb)
quality: etiqueta binaria (0 = saludable, 1 = contaminado). Solo se usa en datos de entrenamiento.

Método obligatorio:

to_vector(): retorna una lista o array con las cuatro medidas [pm25, pm10, o3, no2].



2. Clase AirDataGenerator

Genera datos sintéticos para entrenamiento.

Constructor: __init__(self, num_samples=200) → define cuántas muestras generar.
Método: generate(self) → retorna una lista de objetos AirSample.

Regla para asignar calidad:

if pm25 > 35 or pm10 > 50 or no2 > 40:
    quality = 1  # contaminado
else:
    quality = 0  # saludable
    
Notas importantes:

Para reproducibilidad, fija la semilla de NumPy con np.random.seed(42) dentro del método generate.
Usa np.random.uniform para generar valores aleatorios dentro de los rangos:

pm25: 5 a 100
pm10: 10 a 150
o3: 10 a 100
no2: 5 a 80



3. Clase AirQualityClassifier

Entrena y usa un modelo SVM para clasificar muestras.

Constructor: __init__(self) → crea un modelo SVM (sklearn.svm.SVC) con parámetros por defecto.
Método: fit(self, samples) → recibe una lista de AirSample con calidad definida, y entrena el modelo.
Método: predict(self, sample) → recibe un objeto AirSample sin etiqueta y devuelve la predicción (0 o 1).



4. Clase AirQualityExample

Ejemplo completo de uso.

Método: run(self) que:

Crea un generador AirDataGenerator con 200 muestras.
Genera datos de entrenamiento.
Entrena el clasificador AirQualityClassifier con los datos generados.
Crea una nueva muestra con valores fijos (ejemplo: pm25=22, pm10=30, o3=50, no2=35).
Predice y muestra por pantalla la calidad del aire con un mensaje claro.

"""
# ==============================================================
# 🌍 CLASIFICADOR DE CALIDAD DEL AIRE CON SVM
# --------------------------------------------------------------
# Este programa genera datos sintéticos de contaminación del aire,
# entrena un modelo SVM (Máquina de Vectores de Soporte) y predice
# si una muestra de aire es "saludable" o "contaminada".
# ==============================================================

# Librerías necesarias
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==============================================================
# 1️⃣ CLASE: AirSample
# Representa una muestra individual de aire con sus medidas.
# ==============================================================
class AirSample:
    def __init__(self, pm25, pm10, o3, no2, quality=None):
        """
        Constructor de la clase AirSample.
        Parámetros:
        - pm25: concentración de partículas finas PM2.5 (µg/m³)
        - pm10: concentración de partículas gruesas PM10 (µg/m³)
        - o3: concentración de ozono (ppb)
        - no2: concentración de dióxido de nitrógeno (ppb)
        - quality: etiqueta binaria (0 = saludable, 1 = contaminado)
                   (solo se usa durante el entrenamiento)
        """
        self.pm25 = pm25
        self.pm10 = pm10
        self.o3 = o3
        self.no2 = no2
        self.quality = quality

    def to_vector(self):
        """
        Convierte la muestra en un vector numérico (lista de valores)
        que el modelo de Machine Learning pueda procesar.
        """
        return [self.pm25, self.pm10, self.o3, self.no2]


# ==============================================================
# 2️⃣ CLASE: AirDataGenerator
# Se encarga de generar datos sintéticos (falsos pero realistas)
# para entrenar el modelo.
# ==============================================================
class AirDataGenerator:
    def __init__(self, num_samples=200):
        """
        Constructor.
        num_samples: número de muestras sintéticas que queremos generar.
        """
        self.num_samples = num_samples

    def generate(self):
        """
        Genera las muestras con valores aleatorios en rangos definidos.
        Retorna una lista de objetos AirSample.
        
        """
        np.random.seed(42)  # Semilla fija para reproducibilidad (resultados constantes)
        samples = []

        # Generamos números aleatorios uniformemente distribuidos
        # pm25: 5 a 100, pm10: 10 a 150, o3: 10 a 100, no2: 5 a 80
        pm25_values = np.random.uniform(5, 100, self.num_samples)
        pm10_values = np.random.uniform(10, 150, self.num_samples)
        o3_values = np.random.uniform(10, 100, self.num_samples)
        no2_values = np.random.uniform(5, 80, self.num_samples)

        # Creamos una muestra (AirSample) por cada combinación de valores
        for pm25, pm10, o3, no2 in zip(pm25_values, pm10_values, o3_values, no2_values):
            # Regla para determinar si el aire está contaminado:
            # Si alguno de estos valores supera los límites, se considera contaminado.
            if pm25 > 35 or pm10 > 50 or no2 > 40:
                quality = 1  # Contaminado
            else:
                quality = 0  # Saludable

            # Creamos la muestra y la agregamos a la lista
            samples.append(AirSample(pm25, pm10, o3, no2, quality))

        # Retornamos todas las muestras generadas
        return samples


# ==============================================================
# 3️⃣ CLASE: AirQualityClassifier
# Contiene el modelo SVM que aprenderá a clasificar la calidad del aire.
# ==============================================================
class AirQualityClassifier:
    def __init__(self):
        """
        Constructor.
        Crea un pipeline que incluye:
        - Escalado de datos (StandardScaler)
        - Clasificador SVM con kernel lineal (SVC)
        """
        self.model = make_pipeline(
            StandardScaler(),           # Escalamos los datos
            SVC(kernel='linear', C=1.0, random_state=42)  # Clasificador SVM lineal
        )

    def fit(self, samples):
        """
        Entrena el modelo usando una lista de objetos AirSample.
        """
        # Extraemos las características (X) y etiquetas (y)
        X = [s.to_vector() for s in samples]
        y = [s.quality for s in samples]

        # Entrenamos el modelo
        self.model.fit(X, y)

    def predict(self, sample):
        """
        Predice la calidad del aire para una nueva muestra.
        Retorna:
        - 0: saludable
        - 1: contaminado
        """
        X_new = [sample.to_vector()]  # Convertimos la muestra a formato compatible
        prediction = self.model.predict(X_new)
        return int(prediction[0])


# ==============================================================
# 4️⃣ CLASE: AirQualityExample
# Muestra cómo usar todas las clases juntas en un ejemplo práctico.
# ==============================================================
class AirQualityExample:
    def run(self):
        """
        Método principal: genera datos, entrena el modelo y realiza una predicción.
        """
        print("🌱 Iniciando ejemplo del clasificador de calidad del aire...\n")

        # 1️⃣ Generamos datos sintéticos
        generator = AirDataGenerator(num_samples=200)
        data = generator.generate()
        print(f"✅ {len(data)} muestras de entrenamiento generadas.\n")

        # 2️⃣ Entrenamos el clasificador
        clf = AirQualityClassifier()
        clf.fit(data)
        print("✅ Clasificador SVM entrenado correctamente.\n")
        
        """
        # Usuario introduce valores manualmente  
        
        print("🌍 Introduce los valores de la nueva muestra de aire:")
        pm25 = float(input("➡️  PM2.5 (µg/m³): "))
        pm10 = float(input("➡️  PM10 (µg/m³): "))
        o3   = float(input("➡️  O3 (ppb): "))
        no2  = float(input("➡️  NO2 (ppb): "))

        # Creamos el objeto AirSample con los valores ingresados
        new_sample = AirSample(pm25=pm25, pm10=pm10, o3=o3, no2=no2)

        """

        # 3️⃣ Creamos una nueva muestra de aire para probar el modelo
        new_sample = AirSample(pm25=22, pm10=30, o3=50, no2=35)

        # 4️⃣ Realizamos la predicción
        prediction = clf.predict(new_sample)

        # 5️⃣ Mostramos los resultados
        print("🌍 Muestra de aire:")
        print(f"PM2.5: {new_sample.pm25}, PM10: {new_sample.pm10}, O3: {new_sample.o3}, NO2: {new_sample.no2}")

        if prediction == 0:
            print("✅ Predicción de calidad: Saludable ✅")
        else:
            print("⚠️ Predicción de calidad: Contaminado ⚠️")


# ==============================================================
# 🚀 EJECUCIÓN DEL EJEMPLO
# ==============================================================
if __name__ == "__main__":
    example = AirQualityExample()
    example.run()
