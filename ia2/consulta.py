# prediccion_trastornos.py

import numpy as np
import tensorflow as tf
import joblib
import random
import json



# Constantes
NUM_PREGUNTAS = 35  # Total de preguntas en el cuestionario
NUM_TRASTORNOS = 7  # Número de trastornos a diagnosticar
NOMBRES_TRASTORNOS = ['Ansiedad', 'Depresión', 'Antisocial', 'TDAH', 'Bipolar', 'TOC', 'Misofonía']

# Cargar el modelo y el scaler guardados
model_save = 'ia2/model/modelo_trastornos_mentales_mlp_mejorado3.h5'
scaler_save = 'ia2/scaler/scaler_trastornos_mentales3.pkl'

loaded_model = tf.keras.models.load_model(model_save)
loaded_scaler = joblib.load(scaler_save)

# Umbrales óptimos guardados (deben ser consistentes con los utilizados durante el entrenamiento)
# Asegúrate de haber guardado los umbrales óptimos después del entrenamiento
# Cargar los umbrales óptimos desde el archivo JSON
with open('ia2/optimal_thresholds/optimal_thresholds.json', 'r') as f:
    optimal_thresholds_saved = json.load(f)

# Convertir los umbrales a numpy array si es necesario
optimal_thresholds_saved = np.array(optimal_thresholds_saved)

# Función para realizar predicciones con el modelo cargado
def predecir_trastornos(respuestas):
    """
    Toma un vector de respuestas y predice los trastornos.
    - respuestas: lista o array de 35 elementos con valores 0, 1 o 2
    """
    # Verificar que las respuestas tengan la longitud correcta
    if len(respuestas) != NUM_PREGUNTAS:
        raise ValueError(f"Se esperaban {NUM_PREGUNTAS} respuestas, pero se recibieron {len(respuestas)}.")
    
    # Convertir a numpy array y escalar
    vector_entrada = np.array(respuestas).reshape(1, -1)
    vector_entrada_escalado = loaded_scaler.transform(vector_entrada)
    
    probabilidades = loaded_model.predict(vector_entrada_escalado)[0]
    predicciones_binarias = (probabilidades >= optimal_thresholds_saved).astype(int)
    
    resultados = []
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        resultados.append(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f})")
    return resultados

# Función para generar respuestas variadas
def generar_respuestas_aleatorias():
    respuestas = []
    for i in range(NUM_PREGUNTAS):
        respuestas.append(random.choice([0, 1, 2]))
    return respuestas

# Mostrar el arreglo de respuestas alineado con los resultados de predicción
def mostrar_respuestas_y_predicciones(respuestas, resultados):
    print("\nArreglo de Respuestas:")
    for i in range(0, len(respuestas), 5):
        # Mostrar las respuestas en bloques de 5 para facilitar la lectura
        print(f"{respuestas[i:i+5]}")
    
    print("\nResultados de la Predicción:")
    for resultado in resultados:
        print(resultado)

# Generar varias predicciones con respuestas aleatorias
def generar_multiples_predicciones(n=10):
    for i in range(n):
        print(f"\nPredicción {i+1}:")
        respuestas = generar_respuestas_aleatorias()
        resultados = predecir_trastornos(respuestas)
        mostrar_respuestas_y_predicciones(respuestas, resultados)
        print("-" * 50)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de respuestas del usuario
    # Respuestas codificadas como 0 (No), 1 (A veces), 2 (Sí)
    # Modifica esta lista según las respuestas reales del cuestionario
    generar_multiples_predicciones(n=15)  # Generar 5 predicciones
