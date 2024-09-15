import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, hamming_loss, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os

# ===========================================
# 1. Definición de Constantes
# ===========================================

NUM_PREGUNTAS = 35  # Total de preguntas en el cuestionario
NUM_TRASTORNOS = 7   # Número de trastornos a diagnosticar
PREGUNTAS_POR_TRASTORNO = 5  # Preguntas por trastorno
NOMBRES_TRASTORNOS = ['Ansiedad', 'Depresión', 'Antisocial', 'TDAH', 'Bipolar', 'TOC', 'Misofonía']

# ===========================================
# 2. Generación de Datos Sintéticos
# ===========================================

def generar_datos(num_muestras):
    """
    Genera datos sintéticos para entrenar el modelo.
    
    Cada respuesta se codifica numéricamente: "Sí" = 2, "A veces" = 1, "No" = 0.
    Las etiquetas son porcentajes normalizados (0-1) que indican la probabilidad de presencia de cada trastorno.
    
    Parámetros:
        num_muestras (int): Número de muestras a generar.
        
    Retorna:
        X (np.ndarray): Matriz de respuestas de tamaño (num_muestras, NUM_PREGUNTAS).
        y (np.ndarray): Matriz de etiquetas de tamaño (num_muestras, NUM_TRASTORNOS).
    """
    X = np.zeros((num_muestras, NUM_PREGUNTAS))
    y = np.zeros((num_muestras, NUM_TRASTORNOS))
    
    for i in range(num_muestras):
        # Definir el tipo de caso con diferentes probabilidades
        caso = np.random.choice(['normal', 'single', 'comorbid', 'mixed'], p=[0.2, 0.4, 0.3, 0.1])
        
        if caso == 'normal':
            # Respuestas principalmente "No" con algunas "A veces" y "Sí"
            X[i] = np.random.choice([0, 1, 2], size=NUM_PREGUNTAS, p=[0.75, 0.15, 0.10])
        
        elif caso == 'single':
            # Un solo trastorno positivo
            trastorno = np.random.randint(0, NUM_TRASTORNOS)
            inicio = trastorno * PREGUNTAS_POR_TRASTORNO
            fin = inicio + PREGUNTAS_POR_TRASTORNO
            # Respuestas correlacionadas dentro del trastorno específico
            base_respuesta = np.random.choice([1, 2], p=[0.3, 0.7])
            X[i, inicio:fin] = base_respuesta + np.random.normal(0, 0.2, PREGUNTAS_POR_TRASTORNO).clip(0, 2)
            
            # Respuestas bajas y algo de ruido en las demás preguntas
            mask = np.ones(NUM_PREGUNTAS, dtype=bool)
            mask[inicio:fin] = False
            X[i, mask] = np.random.choice([0, 1, 2], size=np.sum(mask), p=[0.75, 0.2, 0.05]) + np.random.normal(0, 0.1, np.sum(mask)).clip(0, 2)
        
        elif caso == 'comorbid':
            # Comorbilidad: 2 o 3 trastornos positivos con correlación interna
            num_trastornos = np.random.randint(2, 4)
            trastornos = np.random.choice(NUM_TRASTORNOS, size=num_trastornos, replace=False)
            for t in trastornos:
                inicio = t * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                base_respuesta = np.random.choice([1, 2], p=[0.4, 0.6])
                X[i, inicio:fin] = base_respuesta + np.random.normal(0, 0.2, PREGUNTAS_POR_TRASTORNO).clip(0, 2)
            
            # Respuestas bajas y con variabilidad en las demás preguntas
            mask = np.ones(NUM_PREGUNTAS, dtype=bool)
            for t in trastornos:
                inicio = t * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                mask[inicio:fin] = False
            X[i, mask] = np.random.choice([0, 1, 2], size=np.sum(mask), p=[0.8, 0.15, 0.05]) + np.random.normal(0, 0.1, np.sum(mask)).clip(0, 2)
        
        elif caso == 'mixed':
            # Respuestas mezcladas de manera aleatoria pero con cierta estructura de correlación
            for j in range(NUM_TRASTORNOS):
                inicio = j * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                if np.random.rand() > 0.5:
                    X[i, inicio:fin] = np.random.choice([0, 1, 2], size=PREGUNTAS_POR_TRASTORNO, p=[0.5, 0.3, 0.2])
                else:
                    X[i, inicio:fin] = np.random.choice([0, 1, 2], size=PREGUNTAS_POR_TRASTORNO, p=[0.3, 0.4, 0.3])
        
        # Cálculo de etiquetas como porcentajes normalizados (0-1)
        for j in range(NUM_TRASTORNOS):
            inicio = j * PREGUNTAS_POR_TRASTORNO
            fin = inicio + PREGUNTAS_POR_TRASTORNO
            suma_puntos = np.sum(X[i, inicio:fin])
            porcentaje = (suma_puntos / (2 * PREGUNTAS_POR_TRASTORNO)) * 100  # Porcentaje
            y[i, j] = porcentaje / 100.0  # Normalizar a [0, 1]
    
    return X, y

# Generar datos sintéticos
X, y = generar_datos(40000)

# ===========================================
# 3. Verificación de la Distribución de las Etiquetas
# ===========================================

# Visualizar la distribución de etiquetas para cada trastorno
plt.figure(figsize=(20, 15))
for i, nombre in enumerate(NOMBRES_TRASTORNOS):
    plt.subplot(4, 2, i+1)
    sns.histplot(y[:, i] * 100, bins=20, kde=True, color='skyblue')
    plt.title(f'Distribución de Etiquetas - {nombre}')
    plt.xlabel('Porcentaje')
    plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# ===========================================
# 4. División de los Datos en Entrenamiento y Prueba
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================================
# 5. Normalización de los Datos
# ===========================================

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar y transformar los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Transformar los datos de prueba
X_test_scaled = scaler.transform(X_test)

# ===========================================
# 6. Definición de la Arquitectura del Modelo MLP Mejorado
# ===========================================

def crear_modelo_mlp(input_dim, output_dim):
    """
    Crea una red neuronal multicapa (MLP) con múltiples capas ocultas, 
    regularización, normalización por lotes y funciones de activación avanzadas.
    
    Parámetros:
        input_dim (int): Dimensionalidad de la capa de entrada.
        output_dim (int): Dimensionalidad de la capa de salida.
        
    Retorna:
        model (tf.keras.Model): Modelo compilado de la red neuronal.
    """
    model = models.Sequential()
    
    # Primera capa oculta
    model.add(layers.Dense(512, input_dim=input_dim, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    # Segunda capa oculta
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    # Tercera capa oculta
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    
    # Capa de salida
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    
    return model

# Instanciar el modelo
modelo_mlp = crear_modelo_mlp(input_dim=NUM_PREGUNTAS, output_dim=NUM_TRASTORNOS)

# ===========================================
# 7. Compilación del Modelo
# ===========================================

# Compilar el modelo con función de pérdida y métricas adecuadas para regresión
modelo_mlp.compile(optimizer='adam',
                   loss='mean_squared_error',  # Error Cuadrático Medio para regresión
                   metrics=['mean_absolute_error'])  # Error Absoluto Medio

# ===========================================
# 8. Definición de Callbacks para el Entrenamiento
# ===========================================

# Reducción de la tasa de aprendizaje cuando el rendimiento se estanca
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.2, 
                                                 patience=5, 
                                                 min_lr=1e-6,
                                                 verbose=1)

# Detención temprana para prevenir el sobreajuste
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=10, 
                                                  restore_best_weights=True,
                                                  verbose=1)

# ===========================================
# 9. Entrenamiento del Modelo
# ===========================================

history = modelo_mlp.fit(X_train_scaled, y_train, 
                         epochs=100, 
                         batch_size=32,
                         validation_split=0.2, 
                         verbose=1, 
                         callbacks=[reduce_lr, early_stopping])

# ===========================================
# 10. Evaluación del Modelo en el Conjunto de Prueba
# ===========================================

# Evaluar el modelo en el conjunto de prueba
loss, mae = modelo_mlp.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_mlp.predict(X_test_scaled)

# Calcular métricas de regresión
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ===========================================
# 11. Transformación de Predicciones a Etiquetas Binarias (Opcional)
# ===========================================

# Definir umbral para convertir a etiquetas binarias
umbral = 0.5  # Equivale al 50%

# Convertir predicciones y etiquetas reales a binarias
y_pred_binary = (y_pred >= umbral).astype(int)
y_test_binary = (y_test >= umbral).astype(int)

# Calcular métricas de clasificación
hamming = hamming_loss(y_test_binary, y_pred_binary)
f1_micro = f1_score(y_test_binary, y_pred_binary, average='micro')
f1_macro = f1_score(y_test_binary, y_pred_binary, average='macro')

print(f"\nHamming Loss: {hamming:.4f} (Valor óptimo cercano a 0)")
print(f"F1-Score Micro Average: {f1_micro:.4f} (Valor óptimo cercano a 1)")
print(f"F1-Score Macro Average: {f1_macro:.4f} (Valor óptimo cercano a 1)")

global_accuracy = accuracy_score(y_test_binary.flatten(), y_pred_binary.flatten())
print(f"Global Accuracy: {global_accuracy:.4f}")

# Reporte de clasificación detallado
print("\nReporte de Clasificación con Umbral del 50%:")
print(classification_report(y_test_binary, y_pred_binary, target_names=NOMBRES_TRASTORNOS))

# ===========================================
# 12. Visualización de Métricas de Rendimiento
# ===========================================

def plot_metric(history, metric, title, ylabel, ylim=None):
    """
    Función para plotear métricas de entrenamiento y validación.
    
    Parámetros:
        history (History): Historial de entrenamiento del modelo.
        metric (str): Nombre de la métrica a plotear.
        title (str): Título del gráfico.
        ylabel (str): Etiqueta del eje y.
        ylim (tuple, opcional): Límites del eje y.
    """
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

# Gráfico de pérdida (MSE)
plot_metric(history, 'loss', 'Pérdida del Modelo (MSE)', 'MSE')

# Gráfico de Error Absoluto Medio (MAE)
plot_metric(history, 'mean_absolute_error', 'Error Absoluto Medio del Modelo (MAE)', 'MAE')

# ===========================================
# 13. Matriz de Confusión para Cada Trastorno
# ===========================================

for i in range(NUM_TRASTORNOS):
    cm = confusion_matrix(y_test_binary[:, i], y_pred_binary[:, i])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {NOMBRES_TRASTORNOS[i]}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()
    plt.close()

# ===========================================
# 14. Curvas ROC para Cada Trastorno
# ===========================================

plt.figure(figsize=(10, 8))
for i in range(NUM_TRASTORNOS):
    # Utilizar las etiquetas binarias para ROC curve
    fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{NOMBRES_TRASTORNOS[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal de referencia
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC para Cada Trastorno')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
plt.close()

# ===========================================
# 15. Guardado del Modelo y del Escalador
# ===========================================

# Crear directorios si no existen
os.makedirs('ia2/model', exist_ok=True)
os.makedirs('ia2/scaler', exist_ok=True)

# Guardar el modelo entrenado
model_save = 'ia2/model/modelo_trastornos_mentales_mlp_mejorado3v2.h5'
modelo_mlp.save(model_save)
print(f"\nModelo guardado como '{model_save}'")

# Guardar el escalador para futuras predicciones
scaler_save = 'ia2/scaler/scaler_trastornos_mentales3v2.pkl'
joblib.dump(scaler, scaler_save)
print(f"Scaler guardado como '{scaler_save}'")

# ===========================================
# 16. Función para Realizar Predicciones con el Modelo Guardado
# ===========================================

def predecir_trastornos(respuestas, umbral=0.5):
    """
    Toma un vector de respuestas y predice los trastornos.
    
    Parámetros:
        respuestas (list o array): Lista de 35 elementos con valores 0, 1 o 2.
        umbral (float, opcional): Umbral para convertir probabilidades a etiquetas binarias.
        
    Retorna:
        None: Imprime los resultados de la predicción.
    """
    # Verificar que las respuestas tengan la longitud correcta
    if len(respuestas) != NUM_PREGUNTAS:
        raise ValueError(f"Se esperaban {NUM_PREGUNTAS} respuestas, pero se recibieron {len(respuestas)}.")
    
    # Convertir a numpy array y escalar
    vector_entrada = np.array(respuestas).reshape(1, -1)
    vector_entrada_escalado = scaler.transform(vector_entrada)
    
    # Realizar la predicción
    probabilidades_normalizadas = modelo_mlp.predict(vector_entrada_escalado)[0]
    probabilidades = probabilidades_normalizadas * 100  # Desnormalizar a porcentaje
    
    # Convertir probabilidades a etiquetas binarias usando el umbral
    predicciones_binarias = (probabilidades_normalizadas >= umbral).astype(int)
    
    # Mostrar resultados
    print("\nResultados de la Predicción:")
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        print(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f}%)")

# ===========================================
# 17. Ejemplo de Uso de la Función de Predicción
# ===========================================

# Ejemplo: Un individuo responde "Sí" (2) a las primeras 5 preguntas y "No" (0) a las demás
respuestas_ejemplo = [2, 2, 2, 2, 2] + [0]*30
predecir_trastornos(respuestas_ejemplo)

# ===========================================
# 18. Comentarios sobre las Métricas Utilizadas
# ===========================================

"""
Métricas de Regresión:
- Mean Squared Error (MSE): Penaliza más los errores grandes. Valor óptimo: 0.
    - Interpretación: MSE mide el promedio de los errores al cuadrado entre las predicciones y los valores reales. Un MSE más bajo indica un mejor rendimiento del modelo.
- Mean Absolute Error (MAE): Mide el error absoluto promedio. Valor óptimo: 0.
    - Interpretación: MAE calcula la media de las diferencias absolutas entre las predicciones y los valores reales. Es menos sensible a los errores grandes en comparación con MSE.
- R² Score: Indica la proporción de la variabilidad explicada por el modelo. Valor óptimo: 1.
    - Interpretación: R² mide qué tan bien las predicciones del modelo se ajustan a los datos reales. Un R² cercano a 1 indica que el modelo explica la mayor parte de la variabilidad de las etiquetas.

Métricas de Clasificación (Tras binarización):
- Hamming Loss: Proporción de etiquetas incorrectamente predichas. Valor óptimo: 0 (sin errores).
    - Interpretación: Hamming Loss calcula la fracción de etiquetas que están mal clasificadas. Un valor de 0 indica que todas las etiquetas fueron predichas correctamente.
- F1-Score (Micro y Macro): Combina precisión y recall. Valores óptimos: Cercanos a 1.
    - Interpretación:
        - F1-Score Micro: Calcula métricas globales contando el total de verdaderos positivos, falsos negativos y falsos positivos.
        - F1-Score Macro: Calcula métricas para cada clase y luego promedia los resultados. Es útil cuando las clases están desbalanceadas.
- Classification Report: Detalla precisión, recall y F1-score por clase.
    - Interpretación: Proporciona una visión detallada del rendimiento del modelo en cada clase individualmente.

Curvas ROC y AUC:
- ROC Curve (Receiver Operating Characteristic): Grafica la tasa de verdaderos positivos (TPR) contra la tasa de falsos positivos (FPR) a diferentes umbrales.
    - Interpretación: Cuanto más cerca esté la curva ROC del área superior izquierda, mejor es el rendimiento del modelo.
- AUC (Area Under the Curve): Representa el área bajo la curva ROC. Valor óptimo: 1.
    - Interpretación: Un AUC de 0.5 indica un rendimiento aleatorio, mientras que un AUC cercano a 1 indica una excelente capacidad de discriminación del modelo.
Las curvas ROC se han corregido para utilizar etiquetas binarias (`y_test_binary`) en lugar de etiquetas continuas (`y_test`). Esto evita el error "continuous format is not supported" y permite calcular correctamente las métricas ROC para cada trastorno.


Recomendaciones para Interpretar las Métricas:
1. **Balancear las Métricas de Regresión y Clasificación**: Aunque el modelo está optimizado para regresión, las métricas de clasificación proporcionan una perspectiva adicional sobre cómo el modelo está diferenciando entre clases.
2. **Evaluar por Trastorno**: Es importante analizar las métricas para cada trastorno individualmente, ya que algunas clases pueden ser más difíciles de predecir que otras.
3. **Ajustar el Umbral según Necesidad**: El umbral de 0.5 es estándar, pero dependiendo de la aplicación, podrías necesitar ajustarlo para mejorar la precisión o el recall.
4. **Curvas ROC para Entender el Rendimiento**: Utiliza las curvas ROC para visualizar cómo se comporta el modelo en diferentes umbrales y para comparar el rendimiento entre trastornos.
5. **Regularización y Arquitectura**: Si algunas métricas no son satisfactorias, considera ajustar la arquitectura del modelo, los hiperparámetros o las técnicas de regularización.

Este conjunto completo de métricas te proporcionará una visión holística del rendimiento de tu modelo, permitiéndote identificar áreas de mejora y garantizar que el modelo funcione de manera óptima para diferenciar entre los distintos trastornos cognitivos.


Recomendaciones:
1. **Interpretación de Métricas:**
   - **MSE y MAE:** Evalúa qué tan cerca están las predicciones de los valores reales. MSE es más sensible a los errores grandes.
   - **R² Score:** Indica la proporción de variabilidad explicada. Un R² cercano a 1 es deseable.
   - **Hamming Loss y F1-Score:** Evalúa la precisión de las predicciones binarias. Hamming Loss debe ser lo más cercano a 0 posible, mientras que F1-Score debe ser lo más cercano a 1.
   - **Curvas ROC y AUC:** Evalúa la capacidad del modelo para distinguir entre clases. AUC cercano a 1 es óptimo.

2. **Ajuste de Umbral:**
   - El umbral de 0.5 es estándar, pero podrías ajustarlo dependiendo de si prefieres mejorar la precisión o el recall para ciertas clases.

3. **Visualización:**
   - Las matrices de confusión y las curvas ROC proporcionan una visión detallada del rendimiento del modelo por clase, lo que es útil para identificar áreas específicas de mejora.

4. **Regularización y Arquitectura:**
   - Continúa experimentando con diferentes arquitecturas, tasas de dropout y regularización para optimizar el rendimiento y prevenir el sobreajuste.

5. **Validación Cruzada:**
   - Implementa validación cruzada para obtener una evaluación más robusta del rendimiento del modelo y asegurar que no haya sesgo en la partición de los datos.
"""
