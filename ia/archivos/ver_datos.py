from collections import Counter
from itertools import combinations

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

def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

def all_enfermedades():
    X_train = []
    y_train = []
    enfermedades_frecuencia = Counter()

    # Añadir combinaciones de síntomas y enfermedades
    for num_enfermedades in range(1, len(enfermedades) + 1):
        for combo_enfermedades in combinations(enfermedades.keys(), num_enfermedades):
            sintomas_combo = set()
            for enfermedad in combo_enfermedades:
                sintomas_combo.update(enfermedades[enfermedad])
                enfermedades_frecuencia[enfermedad] += 1
            
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

    return X_train, y_train, enfermedades_frecuencia

# Llamar a la función y obtener los resultados
X_train, y_train, enfermedades_frecuencia = all_enfermedades()

# Mostrar la frecuencia de cada enfermedad
print("Frecuencia de cada enfermedad:")
for enfermedad, frecuencia in enfermedades_frecuencia.items():
    print(f"{enfermedad}: {frecuencia} veces")
