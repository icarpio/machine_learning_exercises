"""
Clasificación Automática de Frutas

Contexto:

Eres parte de un equipo que desarrolla una app para supermercados inteligentes. Tu tarea es crear un clasificador automático de frutas basado en dos características:
peso (en gramos) y tamaño (en cm). El modelo debe aprender a distinguir entre Manzanas, Plátanos y Naranjas usando datos generados de forma simulada.

Objetivo:

Construir una solución modular en Python que:

Genere datos aleatorios simulando las características físicas de frutas.
Entrene un clasificador K-Nearest Neighbors (KNN) con esos datos.
Permita predecir el tipo de fruta dados su peso y tamaño.
Muestre gráficamente los datos con colores distintos para cada fruta.

🔧 Especificaciones técnicas

1. Crear la clase GeneradorFrutas

Método: generar(self, num_muestras)

Debe generar num_muestras pares [peso, tamaño] y su respectiva etiqueta: "Manzana", "Plátano" o "Naranja".

Rango de valores por tipo:

Manzana: peso entre 120–200g, tamaño entre 7–9cm
Plátano: peso entre 100–150g, tamaño entre 12–20cm
Naranja: peso entre 150–250g, tamaño entre 8–12cm


2. Crear la clase ClasificadorFrutas

Entrena un modelo KNN y permite hacer predicciones:

Constructor con el parámetro k (número de vecinos).
Método: entrenar(X, y) → divide en entrenamiento/test y ajusta el modelo.
Método: evaluar() → imprime y retorna la precisión del modelo sobre el set de prueba.
Método: predecir(peso, tamaño) → retorna la fruta predicha como string.

3: Crear la clase VisualizadorFrutas

Método: graficar(self, X, y, titulo="Frutas") que grafique un scatter plot (matplotlib), con color distinto por clase.

4: Clase principal SimuladorFrutas

Método: ejecutar(self)

Genera 100 muestras con GeneradorFrutas
Entrena el modelo con ClasificadorFrutas
Predice el tipo de fruta para una muestra nueva: peso 140g y tamaño 18cm
Imprime la predicción.
Muestra un gráfico de las frutas generadas.

✅ Ejemplo de uso

simulador = SimuladorFrutas()
simulador.ejecutar()

Salida esperada

🔍 Precisión del modelo: 90.00%
🍎 La fruta predicha para peso=140g y tamaño=18cm es: Plátano
    
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# ============================================================
# 1. Clase GeneradorFrutas
# ============================================================
class GeneradorFrutas:
    """
    Genera datos simulados de frutas (peso, tamaño) con su etiqueta correspondiente.
    Ahora incluye 8 frutas diferentes.
    """

    def generar(self, num_muestras):
        # cada fruta tendrá el mismo número de muestras
        n = num_muestras // 8

        frutas = {
            "Manzana":  (120, 200, 7, 9),
            "Plátano":  (100, 150, 12, 20),
            "Naranja":  (150, 250, 8, 12),
            "Pera":     (120, 180, 7, 10),
            "Papaya":   (500, 1500, 15, 30),
            "Sandía":   (2000, 9000, 20, 40),
            "Fresa":    (10, 25, 2, 4),
            "Melón":    (800, 3000, 12, 25)
        }

        pesos, tamanos, etiquetas = [], [], []

        # generar datos para cada fruta
        for nombre, (p_min, p_max, t_min, t_max) in frutas.items():
            p = np.random.uniform(p_min, p_max, n)
            t = np.random.uniform(t_min, t_max, n)
            pesos.append(p)
            tamanos.append(t)
            etiquetas.extend([nombre] * n)

        # unir todo en arrays
        pesos = np.concatenate(pesos)
        tamanos = np.concatenate(tamanos)
        etiquetas = np.array(etiquetas)

        X = np.column_stack((pesos, tamanos))
        y = etiquetas

        return X, y



# ============================================================
# 2. CLASE ClasificadorFrutas
# ============================================================
class ClasificadorFrutas:
    """
    Implementa un clasificador KNN para frutas.
    """

    def __init__(self, k=5):
        self.k = k
        self.modelo = KNeighborsClassifier(n_neighbors=k)

    def entrenar(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.modelo.fit(self.X_train, self.y_train)

    def evaluar(self):
        pred = self.modelo.predict(self.X_test)
        acc = accuracy_score(self.y_test, pred)
        print(f"🔍 Precisión del modelo: {acc * 100:.2f}%")
        return acc

    def predecir(self, peso, tamaño):
        entrada = np.array([[peso, tamaño]])
        return self.modelo.predict(entrada)[0]



# ============================================================
# 3. CLASE VisualizadorFrutas
# ============================================================
class VisualizadorFrutas:
    """
    Muestra los datos en un scatter plot con colores por fruta.
    """

    def graficar(self, X, y, titulo="Frutas"):
        plt.figure(figsize=(10, 7))

        frutas_unicas = np.unique(y)
        colores = plt.cm.tab10(np.linspace(0, 1, len(frutas_unicas)))

        for fruta, color in zip(frutas_unicas, colores):
            mask = (y == fruta)
            plt.scatter(X[mask, 0], X[mask, 1], label=fruta, color=color, s=60, edgecolor="black")

        plt.xlabel("Peso (g)")
        plt.ylabel("Tamaño (cm)")
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        plt.show()



# ============================================================
# 4. CLASE PRINCIPAL SimuladorFrutas
# ============================================================
class SimuladorFrutas:
    """
    Ejecuta todo el flujo:
    - Generar datos
    - Entrenar modelo
    - Predecir nueva fruta
    - Graficar
    """

    def ejecutar(self):
        generador = GeneradorFrutas()
        X, y = generador.generar(160)  # 20 muestras por fruta

        clasificador = ClasificadorFrutas(k=5)
        clasificador.entrenar(X, y)

        clasificador.evaluar()

        # Predicción para un dato de ejemplo
        fruta_predicha = clasificador.predecir(140, 18)
        print(f"🍎 La fruta predicha para peso=140g y tamaño=18cm es: {fruta_predicha}")

        visual = VisualizadorFrutas()
        visual.graficar(X, y, "Clasificación de 8 Frutas Simuladas")



# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    simulador = SimuladorFrutas()
    simulador.ejecutar()
