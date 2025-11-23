"""
Predicción de compra de un producto en línea

Contexto:

Eres parte del equipo de análisis de datos en una tienda en línea. Tu objetivo es predecir si un usuario hará una compra o no basándote en su comportamiento en el sitio web. La tienda recopila datos sobre el comportamiento de los usuarios en el sitio, como el número de páginas que visitan y el tiempo que pasan en la página. Te piden que desarrolles un sistema que utilice estos datos para predecir la probabilidad de que un usuario compre un producto.

Objetivo:

Tu tarea es desarrollar un sistema que, a partir de la información sobre el comportamiento de los usuarios, prediga si comprarán o no un producto. 
Debes crear varias funciones que permitan:

Generar un conjunto de datos sintéticos sobre el comportamiento de los usuarios en el sitio.
Entrenar un modelo de predicción basado en el comportamiento de los usuarios.
Realizar predicciones sobre si un nuevo usuario comprará o no el producto en función de su actividad en el sitio web.

Funciones que debes implementar:

Función generar_datos_compras(num_muestras):

Descripción: Esta función debe generar un conjunto de datos sintéticos que representen el comportamiento de los usuarios en el sitio web.
Entrada: Un parámetro num_muestras que indica el número de registros de usuarios que quieres generar.

Proceso:

Cada usuario tendrá dos características:

num_paginas_vistas: Un valor entero entre 1 y 20, que representa el número de páginas que un usuario ha visitado en el sitio.
tiempo_en_sitio: Un valor decimal entre 0 y 30 minutos, que representa el tiempo total que el usuario pasó en el sitio.

La variable objetivo (etiqueta) se asigna como:

1 si el número de páginas vistas es mayor a 5 y el tiempo en el sitio es mayor a 10 minutos.
0 si el número de páginas vistas es 5 o menos o si el tiempo en el sitio es 10 minutos o menos.

Salida: Un par de arrays:

Un array con las características de cada usuario (número de páginas vistas y tiempo en el sitio).
Un array con las etiquetas de compra (1 o 0).
Función graficar_datos(datos, etiquetas)
Descripción: Esta función debe visualizar gráficamente los datos generados para observar cómo se distribuyen los usuarios que compran y los que no compran, en función del número de páginas vistas y el tiempo en el sitio web.

Entrada:

datos: Un array NumPy de tamaño (n, 2) con las características de cada usuario: [número de páginas vistas, tiempo en el sitio].
etiquetas: Un array NumPy de tamaño (n,) con la clase de cada usuario:

1 si el usuario compró,
0 si el usuario no compró.

Proceso:

Crear un gráfico de dispersión con matplotlib:

El eje X representará el número de páginas vistas.
El eje Y representará el tiempo en el sitio (en minutos).

Distinguir los puntos del gráfico según su etiqueta:

Usuarios que compraron: color verde.
Usuarios que no compraron: color rojo.

Añadir elementos gráficos para facilitar la interpretación:

Etiquetas de los ejes.
Título del gráfico.
Leyenda que identifique cada grupo.
Cuadrícula para mejorar la visibilidad.

Salida:

Muestra un gráfico de dispersión que permite analizar visualmente el comportamiento de los usuarios en relación con su decisión de compra.

Función entrenar_modelo(datos):

Descripción: Esta función debe entrenar un modelo de Machine Learning para predecir la compra del usuario basado en las características generadas.
Entrada: El conjunto de datos generado por la función anterior. Debe contener las características (num_paginas_vistas, tiempo_en_sitio) y las etiquetas (0 o 1).

Proceso:

Divide el conjunto de datos en dos partes: un conjunto de entrenamiento y un conjunto de prueba (usualmente 70% entrenamiento, 30% prueba).
Utiliza un algoritmo de clasificación (como la regresión logística) para entrenar el modelo.
Entrena el modelo utilizando las características (num_paginas_vistas y tiempo_en_sitio) y las etiquetas (compra o no compra).
Salida: El modelo entrenado.

Función predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):

Descripción: Esta función debe predecir si un usuario comprará o no el producto basándose en sus características (número de páginas vistas y tiempo en el sitio).

Entrada: El modelo entrenado y las características de un nuevo usuario:

num_paginas_vistas: El número de páginas que un nuevo usuario ha visitado en el sitio.
tiempo_en_sitio: El tiempo que el nuevo usuario ha pasado en el sitio, en minutos.
Salida: Devuelve la predicción de la compra, que puede ser 1 (comprará) o 0 (no comprará).

Función evaluar_modelo(modelo, datos):

Descripción: Esta función debe evaluar el rendimiento del modelo entrenado.
Entrada: El modelo entrenado y el conjunto de datos original (conjunto de características y etiquetas).

Proceso:

Divide el conjunto de datos en un conjunto de entrenamiento y uno de prueba.
Realiza las predicciones sobre el conjunto de prueba.
Calcula y muestra la precisión del modelo, es decir, la proporción de predicciones correctas (puedes usar la métrica de precisión o exactitud).
Salida: Muestra la precisión del modelo y devuelve el valor numérico de precisión.
Función graficar_funcion_prediccion(modelo)

Descripción:
Esta función debe representar gráficamente la probabilidad de que un usuario realice una compra en función del número de páginas vistas, 
manteniendo fijo el tiempo en el sitio web.

Entrada:

modelo: el modelo de regresión logística previamente entrenado.

Proceso:

Define un rango de valores para el número de páginas vistas (por ejemplo, del 1 al 20).
Fija un valor constante para el tiempo en el sitio (por ejemplo, 15 minutos).
Crea un conjunto de datos de entrada donde varíe el número de páginas vistas y el tiempo se mantenga constante.
Usa el modelo para predecir la probabilidad de compra de cada uno de esos usuarios (predict_proba).

Representa los resultados en un gráfico de línea donde:

El eje X representa el número de páginas vistas.
El eje Y representa la probabilidad de compra.
Se debe incluir una rejilla, título, etiquetas de los ejes y una escala de 0 a 1 en el eje Y.

Salida:
Una visualización (gráfico de línea) que muestra cómo varía la probabilidad de que un usuario compre dependiendo de cuántas páginas ha visto, manteniendo constante el tiempo que ha pasado en el sitio.

Ejemplo de uso

# Generamos los datos
datos, etiquetas = generar_datos_compras(100)
 
# Visualizamos los datos
graficar_datos(datos, etiquetas)
 
# Entrenamos el modelo
modelo = entrenar_modelo(datos, etiquetas)
 
# Evaluamos el modelo
evaluar_modelo(modelo, datos, etiquetas)
 
# Predicción de un nuevo usuario
print(predecir_compra(modelo, 8, 12))
 
# Graficar función de predicción
graficar_funcion_prediccion(modelo)

Salida esperada

Precisión del modelo: 0.80
Precisión en el conjunto de prueba: 0.80
'El usuario no comprará el producto.'

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# ============================================================
# 1. FUNCIÓN: Generar datos sintéticos de compra
# ============================================================
def generar_datos_compras(num_muestras):
    """
    Genera un conjunto de datos sintético basado en el comportamiento de usuarios:
    - num_paginas_vistas (1 a 20)
    - tiempo_en_sitio (0 a 30)
    Etiqueta:
    - 1 si páginas > 5 AND tiempo > 10
    - 0 en caso contrario
    """
    
    num_paginas = np.random.randint(1, 21, num_muestras)
    tiempo = np.random.uniform(0, 30, num_muestras)
    
    etiquetas = np.where((num_paginas > 5) & (tiempo > 10), 1, 0)
    
    datos = np.column_stack((num_paginas, tiempo))
    
    return datos, etiquetas



# ============================================================
# 2. FUNCIÓN: Graficar los datos
# ============================================================
def graficar_datos(datos, etiquetas):
    """
    Muestra un scatter plot donde los usuarios compradores aparecen en verde
    y los no-compradores en rojo.
    """
    
    compradores = etiquetas == 1
    no_compradores = etiquetas == 0
    
    plt.figure(figsize=(8, 6))
    plt.scatter(datos[compradores, 0], datos[compradores, 1], color="green", label="Compró")
    plt.scatter(datos[no_compradores, 0], datos[no_compradores, 1], color="red", label="No compró")
    
    plt.xlabel("Número de páginas vistas")
    plt.ylabel("Tiempo en sitio (minutos)")
    plt.title("Comportamiento de usuarios")
    plt.grid(True)
    plt.legend()
    plt.show()



# ============================================================
# 3. FUNCIÓN: Entrenar modelo
# ============================================================
def entrenar_modelo(datos, etiquetas):
    """
    Entrena un modelo de regresión logística para predecir compra.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        datos, etiquetas, test_size=0.3, random_state=42
    )
    
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    
    # evaluación interna en entrenamiento
    pred = modelo.predict(X_train)
    acc = accuracy_score(y_train, pred)
    print(f"Precisión del modelo: {acc:.2f}")
    
    return modelo



# ============================================================
# 4. FUNCIÓN: Predecir compra de un usuario nuevo
# ============================================================
def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    """
    Devuelve 1 si predice compra y 0 si predice no-compra.
    """
    
    entrada = np.array([[num_paginas_vistas, tiempo_en_sitio]])
    pred = modelo.predict(entrada)[0]
    
    if pred == 1:
        return "El usuario comprará el producto."
    else:
        return "El usuario no comprará el producto."



# ============================================================
# 5. FUNCIÓN: Evaluar modelo
# ============================================================
def evaluar_modelo(modelo, datos, etiquetas):
    """
    Evalúa el modelo usando train_test_split y muestra la precisión.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        datos, etiquetas, test_size=0.3, random_state=42
    )
    
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    print(f"Precisión en el conjunto de prueba: {acc:.2f}")
    
    return acc



# ============================================================
# 6. FUNCIÓN: Graficar función de predicción
# ============================================================
def graficar_funcion_prediccion(modelo):
    """
    Muestra la probabilidad de compra en función del número de páginas vistas,
    manteniendo fijo el tiempo en el sitio (ej: 15 minutos).
    """
    
    tiempo_fijo = 15
    paginas = np.arange(1, 21)
    
    datos = np.column_stack((paginas, np.full_like(paginas, tiempo_fijo)))
    
    probabilidades = modelo.predict_proba(datos)[:, 1]
    
    plt.figure(figsize=(8, 5))
    plt.plot(paginas, probabilidades, marker="o")
    
    plt.xlabel("Número de páginas vistas")
    plt.ylabel("Probabilidad de compra")
    plt.title("Probabilidad de compra vs. páginas vistas (tiempo fijo = 15 min)")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()



# ============================================================
# EJEMPLO DE USO COMPLETO
# ============================================================
if __name__ == "__main__":
    
    # 1. Generar datos
    datos, etiquetas = generar_datos_compras(100)
    
    # 2. Graficar
    graficar_datos(datos, etiquetas)
    
    # 3. Entrenar
    modelo = entrenar_modelo(datos, etiquetas)
    
    # 4. Evaluar
    evaluar_modelo(modelo, datos, etiquetas)
    
    # 5. Predecir
    print(predecir_compra(modelo, 8, 12))
    
    # 6. Función de predicción
    graficar_funcion_prediccion(modelo)
