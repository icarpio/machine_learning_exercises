"""
Recomendador de lenguajes de programación

¿Qué lenguaje de programación debo usar?

Imagina que trabajas como consultor de software para una aceleradora de startups.

Tu tarea es desarrollar un sistema inteligente que, en base a ciertas características de un nuevo proyecto tecnológico, recomiende el lenguaje de programación más adecuado.

Para ello, cuentas con un histórico de proyectos clasificados según el lenguaje usado: Python, JavaScript, Java o C++.

Cada proyecto tiene las siguientes características numéricas:

velocidad: qué tan rápido debe ser el desarrollo (0.0 a 1.0)
mantenimiento: importancia del mantenimiento a largo plazo (0.0 a 1.0)
libs: disponibilidad de librerías relevantes para el proyecto (0.0 a 1.0)

tipo_app: tipo de aplicación:

0 = Ciencia de Datos
1 = Aplicación Web
2 = Sistema Embebido
rendimiento: necesidad de alto rendimiento (0.0 a 1.0)

Tu tarea consiste en:

✅ Objetivo

1.- Clase LanguagePredictor

Implementa una clase llamada LanguagePredictor que actúe como un sistema de recomendación de lenguajes de programación. Esta clase debe cumplir con los siguientes requisitos:

En su método __init__, debe:

Crear una instancia del modelo RandomForestClassifier de sklearn.ensemble.
Definir un diccionario label_map que asocie los valores numéricos utilizados como etiquetas con los nombres de los lenguajes de programación:

{
    0: "Python",
    1: "JavaScript",
    2: "Java",
    3: "C++"
}
Debe incluir un método .train(X, y) que:

Reciba dos arreglos de NumPy (X con características y y con etiquetas).
Entrene el modelo de Random Forest con esos datos.
Debe incluir un método .predict(features) que:
Reciba un vector de características (np.ndarray) correspondiente a un nuevo proyecto.
Devuelva el nombre del lenguaje recomendado como una cadena, usando el mapeo definido en label_map.

Esta clase permitirá entrenar un modelo de aprendizaje automático con datos sintéticos y realizar predicciones comprensibles sobre qué lenguaje usar en 
futuros proyectos tecnológicos.



2.- Función generate_dataset(n_samples=100, seed=42)

Implementa una función llamada generate_dataset que genere un conjunto de datos sintético representando distintos proyectos tecnológicos. Esta función debe:

Recibir dos parámetros:

n_samples (entero): número de muestras o proyectos a generar. Por defecto es 100.
seed (entero): semilla para controlar la aleatoriedad y asegurar la reproducibilidad. Por defecto es 42.
Generar, para cada proyecto, un vector de 5 características numéricas aleatorias:
velocidad: qué tan rápido debe desarrollarse el proyecto (valor entre 0.0 y 1.0)
mantenimiento: importancia del mantenimiento a largo plazo (valor entre 0.0 y 1.0)
libs: disponibilidad de librerías relevantes (valor entre 0.0 y 1.0)

tipo_app: tipo de aplicación, representado como un entero aleatorio en el rango [0, 2]:
0: Ciencia de Datos
1: Aplicación Web
2: Sistema Embebido
rendimiento: necesidad de alto rendimiento (valor entre 0.0 y 1.0)

Asignar a cada proyecto una etiqueta numérica correspondiente al lenguaje más adecuado según las siguientes reglas lógicas:

if rendimiento > 0.8 and tipo_app == 2:
    lenguaje = 3  # C++
elif mantenimiento > 0.7 and tipo_app == 1:
    lenguaje = 2  # Java
elif libs > 0.6 and tipo_app == 0:
    lenguaje = 0  # Python
else:
    lenguaje = 1  # JavaScript
Retornar dos objetos numpy.ndarray:

X: matriz con las características de todos los proyectos generados (tamaño n_samples x 5).

y: vector con las etiquetas numéricas (0 a 3) asociadas a cada proyecto, donde:

0: Python
1: JavaScript
2: Java
3: C++

Esta función sirve como generador de datos de entrenamiento para el modelo de predicción de lenguajes.

"""

# ----------------------------------------------------------
# 🧩 Librerías necesarias
# ----------------------------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------
# 🧠 FUNCIÓN: generate_dataset
# ----------------------------------------------------------
def generate_dataset(n_samples=500, seed=42):
    np.random.seed(seed)
    velocidad = np.random.rand(n_samples)
    mantenimiento = np.random.rand(n_samples)
    libs = np.random.rand(n_samples)
    tipo_app = np.random.randint(0, 3, n_samples)
    rendimiento = np.random.rand(n_samples)
    X = np.column_stack([velocidad, mantenimiento, libs, tipo_app, rendimiento])
    y = []

    for i in range(n_samples):
        v, m, l, t, r = velocidad[i], mantenimiento[i], libs[i], tipo_app[i], rendimiento[i]
        if t == 0 and l > 0.7 and r < 0.6:
            lenguaje = 0
        elif t == 1 and v > 0.8 and m < 0.5:
            lenguaje = 1
        elif t == 1 and m > 0.7:
            lenguaje = 2
        elif t == 2 and r > 0.85:
            lenguaje = 3
        elif t == 0 and l > 0.8:
            lenguaje = 4
        elif r > 0.75 and m > 0.6:
            lenguaje = 5
        elif t == 1 and v > 0.7 and l > 0.7:
            lenguaje = 6
        elif t == 2 and r > 0.7 and m < 0.4:
            lenguaje = 7
        elif t == 1 and m > 0.8 and r < 0.6:
            lenguaje = 8
        elif t == 1 and l > 0.7 and m < 0.5:
            lenguaje = 9
        elif t == 0 and r > 0.6 and l < 0.5:
            lenguaje = 10
        elif r > 0.9 and m < 0.5:
            lenguaje = 11
        elif t == 2 and r > 0.8 and m > 0.5:
            lenguaje = 12
        elif t == 1 and v > 0.6 and l > 0.6 and m < 0.6:
            lenguaje = 13
        elif t == 0 and m > 0.8:
            lenguaje = 14
        elif t == 2 and m > 0.7 and r < 0.6:
            lenguaje = 15
        elif t == 0 and l > 0.9:
            lenguaje = 16
        elif t == 1 and l > 0.6 and m > 0.7 and r < 0.6:
            lenguaje = 17
        elif r < 0.3 and m < 0.4:
            lenguaje = 18
        else:
            lenguaje = 19
        y.append(lenguaje)

    return X, np.array(y)


# ----------------------------------------------------------
# 🧠 CLASE: LanguagePredictor
# ----------------------------------------------------------
class LanguagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=120, random_state=42)
        self.label_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++",
            4: "R",
            5: "Rust",
            6: "TypeScript",
            7: "Go",
            8: "Kotlin",
            9: "PHP",
            10: "Julia",
            11: "C",
            12: "C#",
            13: "Ruby",
            14: "Scala",
            15: "Swift",
            16: "MATLAB",
            17: "Dart",
            18: "Perl",
            19: "Elixir"
        }

    def train(self, X, y):
        self.model.fit(X, y)
        print(f"✅ Modelo entrenado con {len(y)} proyectos y {len(self.label_map)} lenguajes.")

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return self.label_map[prediction]


# ----------------------------------------------------------
# 🧪 INTERACCIÓN CON EL USUARIO
# ----------------------------------------------------------
if __name__ == "__main__":
    X, y = generate_dataset(n_samples=1000)
    predictor = LanguagePredictor()
    predictor.train(X, y)

    print("\n🧠 Bienvenido al recomendador de lenguajes de programación")
    print("Introduce las características de tu nuevo proyecto:")
    print("(Usa valores entre 0.0 y 1.0, excepto tipo_app que va de 0 a 2)")
    print("  0 = Ciencia de Datos | 1 = Aplicación Web | 2 = Sistema Embebido")
    print("-" * 60) # Repite el guion 60 veces

    # Función auxiliar para pedir valores validados
    def pedir_valor(nombre, min_val=0.0, max_val=1.0, es_entero=False):
        while True:
            try:
                valor = float(input(f"{nombre} ({min_val}-{max_val}): "))
                if es_entero:
                    valor = int(valor)
                if valor < min_val or valor > max_val:
                    print(f"⚠️ Valor fuera de rango ({min_val}-{max_val}). Intenta de nuevo.")
                else:
                    return valor
            except ValueError:
                print("❌ Entrada no válida. Introduce un número válido.")

    # Pedimos cada característica al usuario
    velocidad = pedir_valor("Velocidad de desarrollo: ")
    mantenimiento = pedir_valor("Importancia del mantenimiento: ")
    libs = pedir_valor("Disponibilidad de librerías: ")
    tipo_app = pedir_valor("Tipo de aplicación (0=Data, 1=Web, 2=Embebido): ", 0, 2, es_entero=True)
    rendimiento = pedir_valor("Necesidad de rendimiento: ")

    # Creamos el vector de entrada
    new_project = np.array([velocidad, mantenimiento, libs, tipo_app, rendimiento])
    
    
    # Ejemplo: proyecto nuevo
    # [velocidad, mantenimiento, libs, tipo_app, rendimiento]
    #new_project = np.array([0.6, 0.8, 0.9, 1, 0.5])

    # Realizamos la predicción
    pred = predictor.predict(new_project)

    print("\n💡 Lenguaje recomendado para tu proyecto:", pred)
