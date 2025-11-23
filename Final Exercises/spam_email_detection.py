"""
Detectar correo electrónico spam
Clasificación de Emails: ¿Spam o No Spam?

Contexto: Tienes un conjunto de datos que contiene información sobre emails. Cada email tiene un conjunto de características, como la longitud del mensaje,
la frecuencia de ciertas palabras clave, la cantidad de enlaces, y otros aspectos relevantes. El objetivo es construir un modelo de clasificación para predecir 
si un email es Spam o No Spam.

Objetivo: Tu tarea es implementar un modelo de clasificación que, dada la información de un email (características como la longitud del mensaje y 
la frecuencia de palabras clave), sea capaz de predecir si el email es Spam (1) o No Spam (0).

Funciones a Implementar:

Generar datos de emails:

Función: generar_datos_emails(num_muestras)

Esta función debe generar un conjunto de datos ficticios con num_muestras emails.

Cada email tendrá las siguientes características:

longitud_mensaje: Un número aleatorio que representa la longitud del email en caracteres (entre 50 y 500).
frecuencia_palabra_clave: Un número aleatorio que representa la frecuencia de una palabra clave relacionada con spam (entre 0 y 1).
cantidad_enlaces: Un número aleatorio que representa la cantidad de enlaces en el email (entre 0 y 10).
Cada email será etiquetado como Spam (1) o No Spam (0).

Entrenar el modelo SVM:

Función: entrenar_modelo_svm(datos, etiquetas)
Esta función debe tomar un conjunto de datos con características de emails y sus etiquetas, y entrenar un modelo de clasificación.

La salida debe ser el modelo entrenado.

Realizar predicciones:

Función: predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces)
Esta función debe tomar un modelo entrenado y las características de un nuevo email, y devolver si el email es Spam o No Spam.
La salida debe ser una cadena de texto que indique si el email es Spam o No Spam.

Instrucciones:

Generar Datos: Para empezar, debes generar un conjunto de datos con emails etiquetados (Spam o No Spam).
Entrenar el Modelo: Entrenar el modelo de clasificación basado en las características del email.
Predicciones: Utiliza el modelo entrenado para predecir si un email es Spam o No Spam según sus características.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# ============================================================
# 1. FUNCIÓN: Generar datos de emails
# ============================================================
def generar_datos_emails(num_muestras):
    """
    Genera un dataset sintético de emails con 3 características:
    - longitud_mensaje:      50 a 500 caracteres
    - frecuencia_palabra:    0 a 1
    - cantidad_enlaces:      0 a 10
    
    Regla para definir SPAM:
    Un email será considerado SPAM (1) si:
      - frecuencia_palabra > 0.5
        o 
      - cantidad_enlaces > 3
    Caso contrario = No Spam (0)
    """

    longitud_mensaje = np.random.randint(50, 501, num_muestras)
    frecuencia_palabra = np.random.uniform(0, 1, num_muestras)
    cantidad_enlaces = np.random.randint(0, 11, num_muestras)

    # Etiqueta sintética basada en comportamiento típico de SPAM
    etiquetas = np.where(
        (frecuencia_palabra > 0.5) | (cantidad_enlaces > 3),
        1,  # spam
        0   # no-spam
    )

    datos = np.column_stack(
        (longitud_mensaje, frecuencia_palabra, cantidad_enlaces)
    )

    return datos, etiquetas


# ============================================================
# 2. FUNCIÓN: Entrenar modelo SVM
# ============================================================
def entrenar_modelo_svm(datos, etiquetas):
    """
    Entrena un modelo de Clasificación SVM para detectar SPAM.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        datos, etiquetas, test_size=0.3, random_state=42
    )

    modelo = SVC(kernel="linear", probability=True)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"🔍 Precisión del modelo SVM: {acc:.2f}")

    return modelo

# ============================================================
# 3. FUNCIÓN: Predecir si un email es spam
# ============================================================
def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
    """
    Usa el modelo entrenado para predecir si un nuevo email es SPAM o NO-SPAM.
    """

    entrada = np.array([[longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]])
    pred = modelo.predict(entrada)[0]

    if pred == 1:
        return "📩 El email ES SPAM."
    else:
        return "📨 El email NO es spam."

# ============================================================
# EJEMPLO DE USO COMPLETO
# ============================================================
if __name__ == "__main__":

    # 1. Generar datos
    datos, etiquetas = generar_datos_emails(500)

    # 2. Entrenar modelo
    modelo = entrenar_modelo_svm(datos, etiquetas)

    # 3. Predicción de ejemplo
    resultado = predecir_email(modelo, 200, 0.7, 5)
    print(resultado)
