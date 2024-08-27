import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from itertools import combinations
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.layers import BatchNormalization, LeakyReLU
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder


scaler = StandardScaler()



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
# Número de síntomas (entradas) y enfermedades (salidas)
# input_dim = 43  # Número de síntomas únicos
# output_dim = 9  # Número de enfermedades

# def crear_modelo(input_dim, output_dim):
#     model = Sequential()
#     model.add(Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
    
#     model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     model.add(LeakyReLU(alpha=0.01))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
    
#     model.add(Dense(output_dim, activation='sigmoid'))
#     model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
#     return model
def crear_modelo(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model
    


def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]


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

def generar_ejemplos_negativos(num_negatives=100):
    X_negativos = []
    y_negativos = []
    
    for _ in range(num_negatives):
        sintomas_aleatorios = random.sample(todos_los_sintomas, random.randint(1, len(todos_los_sintomas)))
        vector_sintomas = crear_vector_sintomas(sintomas_aleatorios)
        vector_enfermedad = [0] * len(enfermedades)
        
        X_negativos.append(vector_sintomas)
        y_negativos.append(vector_enfermedad)
    
    return np.array(X_negativos), np.array(y_negativos)


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

def generar_datos_random(num_ejemplos):
    X_train = []
    y_train = []
    
    # Contar la cantidad de ejemplos por enfermedad
    counts = {enfermedad: 0 for enfermedad in enfermedades}

    for _ in range(num_ejemplos):
        # Elegir cuántas enfermedades incluir
        num_enfermedades = random.randint(1, len(enfermedades))
        combo_enfermedades = random.sample(enfermedades.keys(), num_enfermedades)
        
        # Crear una combinación de síntomas
        sintomas_combo = set()
        for enfermedad in combo_enfermedades:
            sintomas_combo.update(enfermedades[enfermedad])
        
        # Crear vectores binarios
        vector_sintomas = crear_vector_sintomas(sintomas_combo)
        vector_enfermedades = [1 if enfermedad in combo_enfermedades else 0 for enfermedad in enfermedades]
        
        # Añadir datos
        X_train.append(vector_sintomas)
        y_train.append(vector_enfermedades)
        
        # Actualizar el contador de enfermedades
        for enfermedad in combo_enfermedades:
            counts[enfermedad] += 1
        
        # Añadir ejemplos negativos con síntomas que no cumplen criterios completos
        if random.random() < 0.2:
            sintomas_extra = random.sample(todos_los_sintomas, random.randint(1, len(todos_los_sintomas)))
            vector_sintomas_extra = crear_vector_sintomas(sintomas_extra)
            vector_enfermedades_extra = [0] * len(enfermedades)
            
            if random.random() < 0.5:
                X_train.append(vector_sintomas_extra)
                y_train.append(vector_enfermedades_extra)

    # Balancear datos: asegurarse de que no haya sobreajuste a una enfermedad específica
    max_count = max(counts.values())
    balanced_X_train = []
    balanced_y_train = []

    for enfermedad in enfermedades.keys():
        num_examples = counts[enfermedad]
        if num_examples < max_count:
            # Añadir ejemplos adicionales para enfermedades subrepresentadas
            while num_examples < max_count:
                sintomas_combo = set(enfermedades[enfermedad])
                vector_sintomas = crear_vector_sintomas(sintomas_combo)
                vector_enfermedades = [1 if e == enfermedad else 0 for e in enfermedades]
                
                balanced_X_train.append(vector_sintomas)
                balanced_y_train.append(vector_enfermedades)
                num_examples += 1
    
    # Combinar datos originales con datos balanceados
    X_train.extend(balanced_X_train)
    y_train.extend(balanced_y_train)

    return np.array(X_train), np.array(y_train)


def generar_datos():
    # Generar datos de ejemplos combinados
    
    X_train_var, y_train_var = generar_datos_sintomas_variados(num_samples_por_enfermedad=500)
    X_train_sinteticos, y_train_sinteticos = generar_datos_sinteticos(enfermedades, num_samples=200)
    X_train_neg, y_train_neg = generar_ejemplos_negativos(num_negatives=75)
    X_train_bal, y_train_bal = generar_datos_balanceados(num_samples_por_enfermedad=300)
    X_train_all, y_train_all = all_enfermedades()

    
    
    # Combinar todos los datos
    X_train = np.concatenate((X_train_all, X_train_var, X_train_neg,  X_train_bal, X_train_sinteticos), axis=0)
    y_train = np.concatenate((y_train_all, y_train_var, y_train_neg,  y_train_bal, y_train_sinteticos), axis=0)
    # X_train = np.concatenate((X_train_all, X_train_neg,  X_train_bal, X_train_sinteticos), axis=0)
    # y_train = np.concatenate((y_train_all,  y_train_neg,  y_train_bal, y_train_sinteticos), axis=0)
    

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=45)

    # Crear el directorio para guardar los datos si no existe
    directorio_datos = r'ia/archivos/datos_entrenamiento_1'
    os.makedirs(directorio_datos, exist_ok=True)
    
    # Guardar los datos en archivos
    np.savetxt(os.path.join(directorio_datos, 'X_train.txt'), X_train, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'y_train.txt'), y_train, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'X_test.txt'), X_test, fmt='%d')
    np.savetxt(os.path.join(directorio_datos, 'y_test.txt'), y_test, fmt='%d')

    return X_train, X_test, y_train, y_test


def entrenar_modelo():
    X_train, X_test, y_train, y_test = generar_datos()

    model = crear_modelo(len(todos_los_sintomas), len(enfermedades))
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    # Entrenar el modelo
    history = model.fit(X_train, y_train, 
                        epochs=1000, 
                        batch_size=16, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    # Guardar el modelo entrenado
    modelo_guardado = r'ia/modelo/modelo_enfermedades2.h5'
    model.save(modelo_guardado)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # Realizar predicciones
    y_pred = (model.predict(X_test) > 0.5).astype(int)

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

    return model


if __name__ == "__main__":
    entrenar_modelo()

    # Ejemplo de predicción para un nuevo paciente
    
    symptoms = [
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
    modelo_guardado = r'ia/modelo/modelo_enfermedades2.h5'
    model.load_weights(modelo_guardado)
    for nuevo_paciente in symptoms:
    
        # nuevo_paciente = ['nerviosismo', 'fatiga', 'problemas_concentracion', 'irritabilidad']
        vector_paciente = np.array([crear_vector_sintomas(nuevo_paciente)])

        # Hacer predicción
        predicciones = model.predict(vector_paciente)

        # Convertir las predicciones a porcentajes
        percentages = predicciones[0] * 100

        # Mostrar el resultado en formato de diccionario
        resultado = {enfermedad: round(porcentaje, 2) for enfermedad, porcentaje in zip(enfermedades.keys(), percentages)}
        print("Predicciones de las enfermedades con porcentajes de coincidencia:")
        print(resultado)