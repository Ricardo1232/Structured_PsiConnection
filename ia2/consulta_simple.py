# prediccion_trastornos.py

import numpy as np
import tensorflow as tf
import joblib
import json
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Constantes
NUM_PREGUNTAS = 35  # Total de preguntas en el cuestionario
NUM_TRASTORNOS = 7  # Número de trastornos a diagnosticar
NOMBRES_TRASTORNOS = ['Ansiedad', 'Depresión', 'Antisocial', 'TDAH', 'Bipolar', 'TOC', 'Misofonía']

# Cargar el modelo y el scaler guardados
model_save = 'ia2/model/modelo_trastornos_mentales_mlp_mejorado3_god.h5'
scaler_save = 'ia2/scaler/scaler_trastornos_mentales3_god.pkl'

loaded_model = tf.keras.models.load_model(model_save)
loaded_scaler = joblib.load(scaler_save)

# Compilar el modelo con función de pérdida y métricas adecuadas para regresión
loaded_model.compile(optimizer='adam',
                   loss='mean_squared_error',  # Error Cuadrático Medio para regresión
                   metrics=['mean_absolute_error'])  # Error Absoluto Medio

# Umbrales óptimos guardados (deben ser consistentes con los utilizados durante el entrenamiento)
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
    
    # Realizar la predicción
    probabilidades_normalizadas = loaded_model.predict(vector_entrada_escalado)[0]
    probabilidades = probabilidades_normalizadas * 100  # Desnormalizar a porcentaje
    
    # Convertir probabilidades a etiquetas binarias usando el umbral
    predicciones_binarias = (probabilidades_normalizadas >= optimal_thresholds_saved).astype(int)
    
    resultados = []
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        resultados.append(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f})")
    return resultados


# Ejemplo de uso
if __name__ == "__main__":
    # Respuestas codificadas como 0 (No), 1 (A veces), 2 (Sí)
    respuestas = [1,0,0,1,2]*3 +[2]*5 + [1]*5 + [0]*5 + [1]*5
    print(predecir_trastornos(respuestas))
