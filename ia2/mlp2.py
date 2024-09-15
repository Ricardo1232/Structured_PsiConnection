import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, hamming_loss, f1_score,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os

# Constantes
NUM_PREGUNTAS = 35  # Total de preguntas en el cuestionario
NUM_TRASTORNOS = 7  # Número de trastornos a diagnosticar
PREGUNTAS_POR_TRASTORNO = 5  # Preguntas por trastorno
NOMBRES_TRASTORNOS = ['Ansiedad', 'Depresión', 'Antisocial', 'TDAH', 'Bipolar', 'TOC', 'Misofonía']

# Función para generar datos sintéticos con etiquetas binarizadas
def generar_datos(num_muestras):
    X = np.zeros((num_muestras, NUM_PREGUNTAS))
    y = np.zeros((num_muestras, NUM_TRASTORNOS))
    
    for i in range(num_muestras):
        # Incluir más casos positivos
        caso = np.random.choice(['normal', 'single', 'comorbid', 'mixed'], 
                                p=[0.2, 0.4, 0.3, 0.1])
        
        if caso == 'normal':
            # Respuestas principalmente "No"
            X[i] = np.random.choice([0, 1], size=NUM_PREGUNTAS, p=[0.9, 0.1])
        
        elif caso == 'single':
            trastorno = np.random.randint(0, NUM_TRASTORNOS)
            inicio = trastorno * PREGUNTAS_POR_TRASTORNO
            fin = inicio + PREGUNTAS_POR_TRASTORNO
            # Respuestas altas en el trastorno específico
            X[i, inicio:fin] = 2  # "Sí"
            # Respuestas bajas en las demás preguntas
            mask = np.ones(NUM_PREGUNTAS, dtype=bool)
            mask[inicio:fin] = False
            X[i, mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.9, 0.1])
            y[i, trastorno] = 1
        
        elif caso == 'comorbid':
            # Seleccionar entre 2 y 3 trastornos
            num_trastornos = np.random.randint(2, 4)
            trastornos = np.random.choice(NUM_TRASTORNOS, size=num_trastornos, replace=False)
            for t in trastornos:
                inicio = t * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                X[i, inicio:fin] = 2  # "Sí"
                y[i, t] = 1
            # Respuestas bajas en las demás preguntas
            mask = np.ones(NUM_PREGUNTAS, dtype=bool)
            for t in trastornos:
                inicio = t * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                mask[inicio:fin] = False
            X[i, mask] = np.random.choice([0, 1], size=np.sum(mask), p=[0.9, 0.1])
        
        elif caso == 'mixed':
            # Respuestas mezcladas
            X[i] = np.random.choice([0, 1, 2], size=NUM_PREGUNTAS, p=[0.3, 0.4, 0.3])
            # Etiquetar según el puntaje
            for j in range(NUM_TRASTORNOS):
                inicio = j * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                suma_puntos = np.sum(X[i, inicio:fin])
                porcentaje = (suma_puntos / (2 * PREGUNTAS_POR_TRASTORNO)) * 100
                y[i, j] = 1 if porcentaje >= 50 else 0  # Umbral del 50%
        else:
            # En caso de que no se haya asignado etiqueta en 'mixed'
            y[i] = 0
    
    return X, y

# Generar datos sintéticos
X, y = generar_datos(40000)

# Verificar el balance de clases
print("Número de muestras por trastorno:")
for i, nombre in enumerate(NOMBRES_TRASTORNOS):
    print(f"{nombre}: {np.sum(y[:, i])}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos (importante para redes neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(NUM_PREGUNTAS,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_TRASTORNOS, activation='sigmoid')
])
# Compilar el modelo
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])


# Callbacks para ajustar el entrenamiento
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
                    validation_split=0.2, verbose=1, 
                    callbacks=[reduce_lr, early_stopping])

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy (Exact Match Ratio): {accuracy:.4f}")

# Nota: En clasificación multietiqueta, el 'accuracy' estándar (Exact Match Ratio)
# puede ser bajo, ya que requiere que todas las etiquetas de una muestra sean predichas correctamente.
# Es más apropiado utilizar métricas como Hamming Loss y F1-Score.

# Predecir probabilidades en el conjunto de prueba
y_pred_proba = model.predict(X_test_scaled)

# Calcular umbrales óptimos para cada clase utilizando Precision-Recall
# Predecir probabilidades en el conjunto de prueba
y_pred_proba = model.predict(X_test_scaled)

# Calcular umbrales óptimos basados en las curvas ROC
optimal_thresholds = []
for i in range(NUM_TRASTORNOS):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred_proba[:, i])
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmeans)
    optimal_threshold = thresholds[idx]
    optimal_thresholds.append(float(optimal_threshold))  # Convertir a float nativo

# Guardar los umbrales óptimos en un archivo JSON
with open('ia2/optimal_thresholds/optimal_thresholds.json', 'w') as f:
    # Convertir la lista de umbrales a un formato serializable
    json.dump(optimal_thresholds, f)

print("Umbrales óptimos guardados en 'optimal_thresholds.json'")

# Convertir probabilidades a etiquetas binarias usando los umbrales óptimos
y_pred_adjusted = np.zeros_like(y_pred_proba)
for i in range(NUM_TRASTORNOS):
    y_pred_adjusted[:, i] = (y_pred_proba[:, i] >= optimal_thresholds[i]).astype(int)

# Calcular métricas adicionales
# Hamming Loss: Proporción de etiquetas incorrectamente predichas
hamming = hamming_loss(y_test, y_pred_adjusted)
print(f"Hamming Loss: {hamming:.4f} (Valor aceptable cercano a 0)")

# F1-Score Micro y Macro
f1_micro = f1_score(y_test, y_pred_adjusted, average='micro')
f1_macro = f1_score(y_test, y_pred_adjusted, average='macro')
print(f"F1-Score Micro Average: {f1_micro:.4f} (Valor aceptable cercano a 1)")
print(f"F1-Score Macro Average: {f1_macro:.4f} (Valor aceptable cercano a 1)")

# Reporte de clasificación detallado
print("\nReporte de Clasificación con Umbrales Ajustados:")
print(classification_report(y_test, y_pred_adjusted, target_names=NOMBRES_TRASTORNOS))

# Cálculo de precisión global del modelo
# Precisión global entendida como la media de las precisiones por clase
global_precision = np.mean([f1_score(y_test[:, i], y_pred_adjusted[:, i]) for i in range(NUM_TRASTORNOS)])
print(f"Precisión Global del Modelo (F1-Score Medio): {global_precision:.4f}")

# Función para crear gráficos
def plot_metric(history, metric, title, ylabel, ylim=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Entrenamiento')
    plt.plot(history.history[f'val_{metric}'], label='Validación')
    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# Gráficos de pérdida y precisión
plot_metric(history, 'loss', 'Pérdida del Modelo', 'Pérdida')
plot_metric(history, 'accuracy', 'Precisión del Modelo', 'Precisión')

# Matriz de confusión para cada trastorno
for i in range(NUM_TRASTORNOS):
    cm = confusion_matrix(y_test[:, i], y_pred_adjusted[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {NOMBRES_TRASTORNOS[i]}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    plt.close()

# Curvas ROC
plt.figure(figsize=(10, 8))
for i in range(NUM_TRASTORNOS):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{NOMBRES_TRASTORNOS[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para cada trastorno')
plt.legend(loc="lower right")
plt.show()
plt.close()

# Guardar el modelo
model_save = 'ia2/model/modelo_trastornos_mentales_optimizado2.h5'
model.save(model_save)
print(f"\nModelo guardado como '{model_save}'")

# Guardar el scaler para futuras predicciones
import joblib
scaler_save = 'ia2/scaler/scaler_trastornos_mentales.pkl'
joblib.dump(scaler, scaler_save)
print(f"Scaler guardado como '{scaler_save}'")

# =====================================
# Sección para realizar predicciones con el modelo guardado
# =====================================

# Cargar el modelo y el scaler guardados
loaded_model = tf.keras.models.load_model(model_save)
loaded_scaler = joblib.load(scaler_save)

# Umbrales óptimos guardados (deben ser consistentes)
optimal_thresholds_saved = optimal_thresholds  # Asegúrate de guardar y cargar estos valores en un entorno real

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
    probabilidades = loaded_model.predict(vector_entrada_escalado)[0]
    
    # Convertir probabilidades a etiquetas usando los umbrales óptimos
    predicciones_binarias = (probabilidades >= optimal_thresholds_saved).astype(int)
    
    # Mostrar resultados
    print("\nResultados de la predicción:")
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        print(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f})")

# Ejemplo de uso
# Suponiendo que un individuo responde "Sí" (2) a las primeras 5 preguntas y "No" (0) a las demás
respuestas_ejemplo = [2, 2, 2, 2, 2] + [0]*30
predecir_trastornos(respuestas_ejemplo)
