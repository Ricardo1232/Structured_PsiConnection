import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Constantes
NUM_PREGUNTAS = 35
NUM_TRASTORNOS = 7
PREGUNTAS_POR_TRASTORNO = 5
NOMBRES_TRASTORNOS = ['Ansiedad', 'Depresión', 'Antisocial', 'TDAH', 'Bipolar', 'TOC', 'Misofonía']

# Función mejorada para generar datos sintéticos con etiquetas binarias
def generar_datos(num_muestras):
    X = np.zeros((num_muestras, NUM_PREGUNTAS))
    y = np.zeros((num_muestras, NUM_TRASTORNOS))
    
    for i in range(num_muestras):
        caso = np.random.choice(['normal', 'single', 'comorbid', 'subclinical', 'random'],
                                p=[0.4, 0.2, 0.1, 0.2, 0.1])
        
        if caso == 'normal':
            X[i] = np.random.randint(0, 2, NUM_PREGUNTAS)
            # Sin trastornos
            y[i] = 0
        elif caso == 'single':
            trastorno = np.random.randint(0, NUM_TRASTORNOS)
            inicio = trastorno * PREGUNTAS_POR_TRASTORNO
            fin = inicio + PREGUNTAS_POR_TRASTORNO
            X[i, inicio:fin] = np.random.randint(1, 3, PREGUNTAS_POR_TRASTORNO)
            y[i, trastorno] = 1
        elif caso == 'comorbid':
            trastornos = np.random.choice(NUM_TRASTORNOS, size=2, replace=False)
            for t in trastornos:
                inicio = t * PREGUNTAS_POR_TRASTORNO
                fin = inicio + PREGUNTAS_POR_TRASTORNO
                X[i, inicio:fin] = np.random.randint(1, 3, PREGUNTAS_POR_TRASTORNO)
                y[i, t] = 1
        elif caso == 'subclinical':
            X[i] = np.random.randint(0, 2, NUM_PREGUNTAS)
            # Trastornos subclínicos con baja probabilidad
            y[i] = (np.random.rand(NUM_TRASTORNOS) < 0.2).astype(int)
        else:  # 'random'
            X[i] = np.random.randint(0, 3, NUM_PREGUNTAS)
            y[i] = (np.random.rand(NUM_TRASTORNOS) < 0.5).astype(int)
    
    return X, y

# Generar datos
X, y = generar_datos(30000)

# Verificar el balance de clases
print("Número de muestras por trastorno:")
for i, nombre in enumerate(NOMBRES_TRASTORNOS):
    print(f"{nombre}: {np.sum(y[:, i])}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo mejorado
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(NUM_PREGUNTAS,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_TRASTORNOS, activation='sigmoid')
])

# Compilar el modelo con optimizador Adam y tasa de aprendizaje ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks para reducción de tasa de aprendizaje y early stopping
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
                    validation_split=0.2, verbose=1, 
                    callbacks=[reduce_lr, early_stopping])

# Evaluar el modelo
loss, accuracy, auc_metric = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc_metric:.4f}")

# Predecir probabilidades
y_pred_proba = model.predict(X_test_scaled)

# Calcular umbrales óptimos para cada clase
optimal_thresholds = []
for i in range(NUM_TRASTORNOS):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred_proba[:, i])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_thresholds.append(optimal_threshold)

# Convertir probabilidades a etiquetas usando los umbrales óptimos
y_pred_adjusted = np.zeros_like(y_pred_proba)
for i in range(NUM_TRASTORNOS):
    y_pred_adjusted[:, i] = (y_pred_proba[:, i] >= optimal_thresholds[i]).astype(int)

# Reporte de clasificación
print("\nReporte de Clasificación con Umbrales Ajustados:")
print(classification_report(y_test, y_pred_adjusted, target_names=NOMBRES_TRASTORNOS))

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
model_save = 'ia2/model/modelo_trastornos_mentales_optimizado.h5'
model.save(model_save)
print(f"\nModelo guardado como '{model_save}'")
