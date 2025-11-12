
"""
Clasificar piezas industriales

🎯 Objetivo

Desarrollar un sistema automático de inspección de calidad para piezas industriales usando aprendizaje automático. 
Para ello, implementarás varias clases en Python que simulan la creación de datos, entrenan un modelo de clasificación (SVM) y visualizan los resultados.

Este proyecto se validará con tests automáticos, por lo tanto, las clases y métodos deben tener exactamente los nombres y firmas indicadas.



📦 Clases y métodos obligatorios

1. class Piece

Representa una pieza individual con sus características físicas y su etiqueta de calidad.

Constructor:

def __init__(self, texture, symmetry, edges, center_offset, label):

texture (float): Nivel de textura/homogeneidad (entre 0 y 1).
symmetry (float): Grado de simetría (entre 0 y 1).
edges (float): Número de bordes detectados.
center_offset (float): Desviación del centro respecto al ideal.
label (str): "Correcta" o "Defectuosa".

Método requerido:

def to_vector(self) -> list:
    # Devuelve [texture, symmetry, edges, center_offset]

2. class PieceDatasetGenerator

Genera una lista de objetos Piece simulando datos industriales con una lógica de clasificación basada en reglas.

Constructor:

def __init__(self, n=400):

n: número de piezas a generar (por defecto 400).

Método requerido:

def generate(self) -> list:
    # Devuelve una lista de objetos Piece, cada uno con su etiqueta calculada
💡 Lógica de generación:

Cada característica se genera aleatoriamente según distribuciones normales:


🔎 Reglas de clasificación:

Una pieza será etiquetada como "Defectuosa" si cumple al menos una de estas condiciones:

symmetry < 0.4 y center_offset > 0.25,
o bien texture < 0.35,
o bien edges < 30,
o bien center_offset > 0.35.

En caso contrario, será etiquetada como "Correcta"



3. class PieceClassifier

Entrena un modelo de clasificación usando SVM y permite evaluar y predecir etiquetas de nuevas piezas.

Constructor:

def __init__(self):

Métodos requeridos:

def fit(self, pieces: list) -> None:
    # Entrena el modelo SVM con una lista de objetos Piece
def predict(self, texture, symmetry, edges, center_offset) -> str:
    # Predice si una pieza con esas características es "Correcta" o "Defectuosa"
def evaluate(self, test_data: list) -> None:
    # Muestra matriz de confusión e informe de clasificación (usa sklearn)
El modelo debe usar:

from sklearn.svm import SVC
SVC(kernel='rbf', gamma='scale', C=1.0)


4. class PieceAnalysisExample

Clase demostrativa que conecta todas las partes del proyecto y muestra un ejemplo completo de uso del sistema.

Método requerido:

def run(self) -> None:

Este método debe realizar todo el flujo de trabajo del sistema:

✅ Flujo completo requerido:

Generación de datos:

Crear un objeto PieceDatasetGenerator (usar valor por defecto: 400 piezas).
Llamar a .generate() para obtener las piezas.

División de datos:

Usar train_test_split de sklearn.model_selection.
Separar en 70% entrenamiento y 30% test.
Usar random_state=42.

Entrenamiento:

Crear un PieceClassifier.
Llamar a .fit() con los datos de entrenamiento.

Evaluación:

Llamar a .evaluate() con los datos de prueba.
Mostrar matriz de confusión e informe de clasificación.

Predicción personalizada:

Predecir la clase de una pieza con estas características:

(0.45, 0.5, 45, 0.15)

Mostrar por pantalla las características y el resultado predicho.
Visualización:
Crear un DataFrame con los siguientes campos:

"Textura", "Simetría", "Bordes", "Offset", "Etiqueta"

Crear un scatter plot:

Eje X: "Textura"
Eje Y: "Offset"
Colores: verde = "Correcta", rojo = "Defectuosa"
Agregar título: "🏭 Clasificación de piezas industriales"
Mostrar leyenda y rejilla
"""



# -----------------------------
# 🏭 Clasificación de Piezas Industriales
# -----------------------------

# Librerías que necesitamos
import random           # Para generar números aleatorios
import matplotlib.pyplot as plt  # Para hacer gráficos
import pandas as pd     # Para manejar tablas de datos
from sklearn.svm import SVC   # Para crear nuestro clasificador
from sklearn.model_selection import train_test_split  # Para separar datos en entrenamiento y prueba
from sklearn.metrics import confusion_matrix, classification_report  # Para evaluar el modelo

# -----------------------------
# 1️⃣ Cada pieza es como un "objetito" con características
# -----------------------------
class Piece:
    def __init__(self, texture, symmetry, edges, center_offset, label):
        # Guardamos las características de la pieza
        self.texture = texture          # Qué tan "lisa" o "texturizada" es
        self.symmetry = symmetry        # Qué tan simétrica está
        self.edges = edges              # Cuántos bordes tiene
        self.center_offset = center_offset  # Cuánto se desvió del centro ideal
        self.label = label              # "Correcta" o "Defectuosa"

    def to_vector(self) -> list:
        # Convertimos las características a una lista para el modelo
        return [self.texture, self.symmetry, self.edges, self.center_offset]

# -----------------------------
# 2️⃣ Generador de piezas aleatorias
# -----------------------------
class PieceDatasetGenerator:
    def __init__(self, n=400):
        self.n = n  # Cuántas piezas queremos generar

    def generate(self) -> list:
        pieces = []
        
        """
        Por qué usamos .gauss para generar características de piezas?

        En la simulación de datos industriales:
        Las piezas no siempre tienen valores exactos, sino que tienen variaciones naturales.
        Por ejemplo, la simetría de una pieza rara vez es exactamente 0.5; puede estar cerca de 0.5, pero con pequeñas desviaciones.
        Usar gauss permite simular variaciones realistas, en lugar de valores totalmente aleatorios y uniformes.
        
        Si hubieras usado random.uniform(a, b):

        Todos los valores serían equiprobables entre a y b.
        No reflejaría que la mayoría de piezas tienen valores cercanos a la media y solo unas pocas están lejos.
        random.gauss es clave para entrenar modelos de ML realistas, porque simula el ruido natural de los datos industriales.
        """
        for _ in range(self.n):
            # 🎲 Generamos valores aleatorios "normales" para cada característica
            texture = min(max(random.gauss(0.5, 0.15), 0), 1)
            symmetry = min(max(random.gauss(0.5, 0.2), 0), 1)
            edges = max(int(random.gauss(40, 10)), 0)
            center_offset = min(max(random.gauss(0.2, 0.1), 0), 1)

            # 🛠 Reglas para decidir si la pieza es buena o mala
            if (symmetry < 0.4 and center_offset > 0.25) or texture < 0.35 or edges < 30 or center_offset > 0.35:
                label = "Defectuosa"
            else:
                label = "Correcta"

            # Creamos la pieza y la guardamos
            piece = Piece(texture, symmetry, edges, center_offset, label)
            pieces.append(piece)
        return pieces

# -----------------------------
# 3️⃣ Clasificador SVM
# -----------------------------
class PieceClassifier:
    def __init__(self):
        # Creamos el modelo SVM (es como un juez que decide si la pieza es buena o mala)
        self.model = SVC(kernel='rbf', gamma='scale', C=1.0)

    def fit(self, pieces: list) -> None:
        # Entrenamos al "juez" usando nuestras piezas de entrenamiento
        X = [p.to_vector() for p in pieces]  # Características
        y = [p.label for p in pieces]        # Etiquetas
        self.model.fit(X, y)

    def predict(self, texture, symmetry, edges, center_offset) -> str:
        # Preguntamos al juez sobre una pieza nueva
        X_new = [[texture, symmetry, edges, center_offset]]
        return self.model.predict(X_new)[0]

    def evaluate(self, test_data: list) -> None:
        # Comprobamos qué tan bueno es nuestro juez
        X_test = [p.to_vector() for p in test_data]
        y_test = [p.label for p in test_data]
        y_pred = self.model.predict(X_test)

        print("\n📊 Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_test, y_pred))

# -----------------------------
# 4️⃣ Clase de ejemplo: conecta todo
# -----------------------------
class PieceAnalysisExample:
    def run(self) -> None:
        # 1️⃣ Generamos piezas
        print("🎉 Generando piezas...")
        generator = PieceDatasetGenerator()
        pieces = generator.generate()

        # 2️⃣ Dividimos en entrenamiento y prueba
        # - 70% de los datos para entrenar (train)
        # - 30% de los datos para probar (test)
        # - random_state=42 para que siempre salga lo mismo y por lo tanto sea reproducible
        print("📚 Separando datos en entrenamiento y prueba...")
        train_pieces, test_pieces = train_test_split(pieces, test_size=0.3, random_state=42)

        # 3️⃣ Entrenamos al clasificador
        print("🤖 Entrenando clasificador...")
        classifier = PieceClassifier()
        classifier.fit(train_pieces)

        # 4️⃣ Evaluamos el clasificador
        print("✅ Evaluando clasificador...")
        classifier.evaluate(test_pieces)

        # 5️⃣ Probamos con una pieza específica
        texture, symmetry, edges, offset = 0.45, 0.5, 45, 0.15
        prediction = classifier.predict(texture, symmetry, edges, offset)
        print("\n🔎 Predicción de pieza personalizada:")
        print(f"  → Textura: {texture}, Simetría: {symmetry}, Bordes: {edges}, Offset: {offset}")
        print(f"  → Clasificación: {prediction}")

        # 6️⃣ Visualización simple
        print("📊 Mostrando gráfico de Textura vs Offset...")
        df = pd.DataFrame([{
            "Textura": p.texture,
            "Simetría": p.symmetry,
            "Bordes": p.edges,
            "Offset": p.center_offset,
            "Etiqueta": p.label
        } for p in pieces])

        colors = df['Etiqueta'].map({'Correcta':'green', 'Defectuosa':'red'})
        plt.figure(figsize=(8,6))
        plt.scatter(df['Textura'], df['Offset'], c=colors)
        plt.xlabel("Textura")
        plt.ylabel("Offset")
        plt.title("🏭 Clasificación de piezas industriales")
        plt.grid(True)
        # Leyenda con colores
        plt.legend(handles=[plt.Line2D([0],[0], marker='o', color='w', label='Correcta',
                                       markerfacecolor='green', markersize=10),
                            plt.Line2D([0],[0], marker='o', color='w', label='Defectuosa',
                                       markerfacecolor='red', markersize=10)])
        plt.show()

# -----------------------------
# EJECUTAMOS EL EJEMPLO
# -----------------------------
example = PieceAnalysisExample()
example.run()
