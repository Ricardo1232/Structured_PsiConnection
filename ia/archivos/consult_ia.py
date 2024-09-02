from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU,Input
import numpy as np

def crear_modelo(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(86, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.02)))

    
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


# Función para crear un vector de síntomas
def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]


if __name__ == "__main__":

    symptoms = [
        #completos
    # [ "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno"],
    # ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso",  "problemas_sueno", "fatiga", "pensamientos_suicidio"],
    # [ "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"],
    # ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos",  "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"],
    # [ "perdida_memoria", "dificultad_palabras_conversaciones",  "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento",  "dificultad_tareas_cotidianas"],
    # ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],
    # ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"],
    # ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    # [ "desprecio_normas_sociales", "manipulacion_engano",  "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"],
        
        
        
    # ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad"],  # Ejemplo con síntomas de ansiedad
    # ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso"],  # Ejemplo con síntomas de depresión
    # ["dificultad_atencion", "hiperactividad", "impulsividad"],  # Ejemplo con síntomas de TDAH
    # ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos"],  # Ejemplo con síntomas de Parkinson
    # ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento"],  # Ejemplo con síntomas de Alzheimer
    # ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],  # Ejemplo con síntomas de trastorno bipolar
    # ["obsesiones", "compulsiones"],  # Ejemplo con síntomas de TOC
    # ["irritabilidad", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],  # Ejemplo con síntomas de misofonía
    # ["desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"],  # Ejemplo con síntomas de trastorno antisocial
    # ['nerviosismo', 'fatiga', 'problemas_concentracion', 'irritabilidad'],
    # ['nerviosismo', 'fatiga', 'problemas_concentracion', 'irritabilidad', "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes", "desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"]
    ["preocupacion_excesiva","problemas_concentracion", "sentimientos_tristeza", "perdida_interes" ]
    ]
    model = crear_modelo(43, 9)
    modelo_guardado = r'ia/modelo/modelo_enfermedades3 - god.h5'
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