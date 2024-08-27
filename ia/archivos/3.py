import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU,Input
from tensorflow.keras.optimizers import Adam
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

scaler = StandardScaler()

# Definición de enfermedades y síntomas
enfermedades = {
    "trastorno_ansiedad": [
        "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", 
        "irritabilidad", "tension_muscular", "problemas_sueno"
    ],
    "depresion": [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio"
    ],
    "tdah": [
        "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"
    ],
    "parkinson": [
        "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", 
        "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"
    ],
    "alzheimer": [
        "perdida_memoria", "dificultad_palabras_conversaciones", 
        "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", 
        "dificultad_tareas_cotidianas"
    ],
    "trastorno_bipolar": [
        "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"
    ],
    "toc": [
        "obsesiones", "compulsiones", "reconocimiento_ineficacia_control"
    ],
    "misofonia": [
        "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", 
        "necesidad_escapar", "sonidos_desencadenantes"
    ],
    "trastorno_antisocial": [
        "desprecio_normas_sociales", "manipulacion_engano", 
        "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", 
        "incapacidad_relaciones_estables"
    ]
}

todos_los_sintomas = [
    "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno",  # trastorno_ansiedad
    "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "pensamientos_suicidio",  # depresion
    "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones",  # tdah
    "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir",  # parkinson
    "perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas",  # alzheimer
    "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad",  # trastorno_bipolar
    "obsesiones", "compulsiones", "reconocimiento_ineficacia_control",  # toc
    "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes",  # misofonia
    "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"  # trastorno_antisocial
]

# Función para crear el modelo
# def crear_modelo(input_dim, output_dim):
#     model = Sequential()
    
#     # Capa de entrada
#     model.add(Input(shape=(input_dim,)))
#     model.add(Dense(86, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
#     # model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
    
#     # # Capa oculta 1
#     model.add(Dense(86, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
#     # model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
    
    
#     # Capa oculta 2
#     model.add(Dense(172, activation='relu'))
#     # model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
    
#     # Capa oculta 3
#     model.add(Dense(86, activation='relu'))
#     # model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
    
#     # Capa de salida
#     model.add(Dense(output_dim, activation='sigmoid'))
    
#     # Compilación del modelo
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
#     return model

def crear_modelo(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(86, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    
    model.add(Dense(86, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(86, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0009), loss='binary_crossentropy', metrics=['accuracy'])
    return model
    


# Función para crear un vector de síntomas
def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]
# Función para generar datos con síntomas variados
def generar_datos_sintomas_variados(num_samples_por_enfermedad=100):
    X_train = []
    y_train = []
    
    for enfermedad, sintomas in enfermedades.items():
        for _ in range(num_samples_por_enfermedad):
            # Generar un subconjunto aleatorio de síntomas
            num_sintomas = random.randint(1, len(sintomas))
            sintomas_seleccionados = random.sample(sintomas, num_sintomas)
            
            # Crear vectores binarios para los síntomas y la enfermedad
            vector_sintomas = crear_vector_sintomas(sintomas_seleccionados)
            vector_enfermedad = [1 if e == enfermedad else 0 for e in enfermedades]
            
            # Añadir el vector a los datos de entrenamiento
            X_train.append(vector_sintomas)
            y_train.append(vector_enfermedad)
    
    return np.array(X_train), np.array(y_train)

def generar_datos_sinteticos(enfermedades, num_samples=300):
    sintomaticos = []
    etiquetas = []

    for _ in range(num_samples):
        enfermedades_actuales = random.sample(list(enfermedades.keys()), random.randint(1, len(enfermedades)))
        
        sintomas_variados = set()
        for enfermedad in enfermedades_actuales:
            sintomas_variados.update(enfermedades[enfermedad])
        
        sintomaticos.append(crear_vector_sintomas(sintomas_variados))
        etiquetas.append([1 if e in enfermedades_actuales else 0 for e in enfermedades.keys()])

    return np.array(sintomaticos), np.array(etiquetas)

# Función para generar ejemplos negativos
def generar_ejemplos_negativos(num_negativos=100):
    X_negativos = []
    y_negativos = []
    
    for _ in range(num_negativos):
        sintomas_aleatorios = random.sample(todos_los_sintomas, random.randint(1, len(todos_los_sintomas)))
        vector_sintomas = crear_vector_sintomas(sintomas_aleatorios)
        vector_enfermedad = [0] * len(enfermedades)
        
        X_negativos.append(vector_sintomas)
        y_negativos.append(vector_enfermedad)
    
    return np.array(X_negativos), np.array(y_negativos)

# Función para generar datos balanceados
def generar_datos_balanceados(num_samples_por_enfermedad=200):
    X_train = []
    y_train = []
    
    for enfermedad, sintomas in enfermedades.items():
        for _ in range(num_samples_por_enfermedad):
            # Crear un vector binario para los síntomas de la enfermedad
            vector_sintomas = crear_vector_sintomas(sintomas)
            vector_enfermedad = [1 if e == enfermedad else 0 for e in enfermedades]
            
            # Añadir el vector a los datos de entrenamiento
            X_train.append(vector_sintomas)
            y_train.append(vector_enfermedad)
    
    return np.array(X_train), np.array(y_train)

# Función para generar datos combinados de todas las enfermedades
def all_enfermedades():
    X_train = []
    y_train = []

    # Añadir combinaciones de síntomas y enfermedades
    for num_enfermedades in range(1, len(enfermedades) + 1):
        for combo_enfermedades in combinations(enfermedades.keys(), num_enfermedades):
            sintomas_combo = set()
            for enfermedad in combo_enfermedades:
                sintomas_combo.update(enfermedades[enfermedad])
            
            # Crear un vector binario para los síntomas combinados
            vector_sintomas = crear_vector_sintomas(sintomas_combo)
            
            # Crear un vector binario para las enfermedades combinadas
            vector_enfermedades = [1 if enfermedad in combo_enfermedades else 0 for enfermedad in enfermedades]
            
            # Agregar los vectores a los datos de entrenamiento
            X_train.append(vector_sintomas)
            y_train.append(vector_enfermedades)

    # Añadir el caso de 0 en todo
    X_train.append([0] * len(todos_los_sintomas))
    y_train.append([0] * len(enfermedades))

    return np.array(X_train), np.array(y_train)

# Modificación de la función generar_datos para incluir los nuevos datos
def generar_datos():
    X_train_var, y_train_var                = generar_datos_sintomas_variados(num_samples_por_enfermedad=300)
    X_train_sinteticos, y_train_sinteticos  = generar_datos_sinteticos(enfermedades, num_samples=700)
    # X_train_negativos, y_train_negativos    = generar_ejemplos_negativos(num_negativos=50)
    X_train_balanceado, y_train_balanceado  = generar_datos_balanceados(num_samples_por_enfermedad=300)
    X_train_combo, y_train_combo            = all_enfermedades()
    
    
    # X_train = np.vstack([X_train_sinteticos, X_train_var, X_train_negativos, X_train_balanceado, X_train_combo])
    # y_train = np.vstack([y_train_sinteticos, y_train_var, y_train_negativos, y_train_balanceado, y_train_combo])
    X_train = np.vstack([X_train_sinteticos, X_train_balanceado, X_train_combo, X_train_var])
    y_train = np.vstack([y_train_sinteticos, y_train_balanceado, y_train_combo, y_train_var])
    
    # Normalización de los datos
    X_train = scaler.fit_transform(X_train)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=43)
    
    directorio_datos = r'ia/archivos/datos_entrenamiento_1'
    os.makedirs(directorio_datos, exist_ok=True)
    
    np.savetxt(os.path.join(directorio_datos, 'X_train.txt'), X_train, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'y_train.txt'), y_train, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'X_test.txt'), X_test, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'y_test.txt'), y_test, fmt='%d')

    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo con validación cruzada

# Función para entrenar el modelo con validación cruzada
def entrenar_modelo():
    epochs = 1000
    batch_size = 16
    
    X_train, X_test, y_train, y_test = generar_datos()
    modelo = crear_modelo(input_dim=len(todos_los_sintomas), output_dim=len(enfermedades))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=43)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        print(f"Entrenando fold {fold + 1}/{kf.get_n_splits()}")
        history = modelo.fit(X_train_fold, y_train_fold, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_val_fold, y_val_fold),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        
        # y_pred_val = (modelo.predict(X_val_fold) > 0.5).astype(int)
        # cm = confusion_matrix(y_val_fold.argmax(axis=1), y_pred_val.argmax(axis=1))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title(f"Confusión Fold {fold + 1}")
        # plt.show()
    
    # Guardar el modelo
    modelo_guardado = r'ia/modelo/modelo_enfermedades3.h5'
    modelo.save(modelo_guardado)
    
    loss, accuracy = modelo.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Realizar predicciones
    y_pred = (modelo.predict(X_test) > 0.5).astype(int)

    # Calcular y mostrar la matriz de confusión
    y_test_max = y_test.argmax(axis=1)
    y_pred_max = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred

    cm = confusion_matrix(y_test_max, y_pred_max, labels=range(len(enfermedades)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enfermedades.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Mostrar el historial de entrenamiento
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('Accuracy over Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, max(history.history['loss'])])
    plt.legend(loc='upper right')
    plt.title('Loss over Epochs')
    
    plt.show()




if __name__ == "__main__":
    print(len(todos_los_sintomas))

    entrenar_modelo()

    # Ejemplo de predicción para un nuevo paciente
    
    symptoms = [
        #completos
    [ "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno"],
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso",  "problemas_sueno", "fatiga", "pensamientos_suicidio"],
    [ "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"],
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos",  "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"],
    [ "perdida_memoria", "dificultad_palabras_conversaciones",  "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento",  "dificultad_tareas_cotidianas"],
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"],
    ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    [ "desprecio_normas_sociales", "manipulacion_engano",  "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"],
        
        
        
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad"],  # Ejemplo con síntomas de ansiedad
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso"],  # Ejemplo con síntomas de depresión
    ["dificultad_atencion", "hiperactividad", "impulsividad"],  # Ejemplo con síntomas de TDAH
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos"],  # Ejemplo con síntomas de Parkinson
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento"],  # Ejemplo con síntomas de Alzheimer
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],  # Ejemplo con síntomas de trastorno bipolar
    ["obsesiones", "compulsiones"],  # Ejemplo con síntomas de TOC
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],  # Ejemplo con síntomas de misofonía
    ["desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"],  # Ejemplo con síntomas de trastorno antisocial
    ['nerviosismo', 'fatiga', 'problemas_concentracion', 'irritabilidad'],
    ['nerviosismo', 'fatiga', 'problemas_concentracion', 'irritabilidad', "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes", "desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"]
    ]
    model = crear_modelo(len(todos_los_sintomas), len(enfermedades))
    modelo_guardado = r'ia/modelo/modelo_enfermedades3.h5'
    model.load_weights(modelo_guardado)
    for nuevo_paciente in symptoms:
    
        vector_paciente = np.array([crear_vector_sintomas(nuevo_paciente)])

        # Hacer predicción
        predicciones = model.predict(vector_paciente)

        # Convertir las predicciones a porcentajes
        percentages = predicciones[0] * 100

        # Mostrar el resultado en formato de diccionario
        resultado = {enfermedad: round(porcentaje, 2) for enfermedad, porcentaje in zip(enfermedades.keys(), percentages)}
        print(nuevo_paciente)
        print("Predicciones de las enfermedades con porcentajes de coincidencia:")
        print(resultado, "\n")