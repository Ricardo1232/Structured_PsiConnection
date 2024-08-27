import numpy as np

# Listado completo de síntomas y enfermedades
todos_los_sintomas = [
    "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", 
    "tension_muscular", "problemas_sueno", "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso",
    "pensamientos_suicidio", "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones",
    "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", 
    "dificultad_hablar_escribir", "perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal",
    "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas", "episodios_mania", "episodios_depresion",
    "cambios_bruscos_humor_actividad", "obsesiones", "compulsiones", "reconocimiento_ineficacia_control", 
    "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", 
    "sonidos_desencadenantes", "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", 
    "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"
]

etq_enfermedades = [
    "trastorno_ansiedad", "depresion", "tdah", "parkinson", "alzheimer", 
    "trastorno_bipolar", "toc", "misofonia", "trastorno_antisocial"
]

# Inicializar listas para los datos de entrada (X) y etiquetas (y)
X = []
y = []

# Convertir los diccionarios en arrays
for ejemplo in datos_entrenamiento:
    # Crear la lista para los síntomas
    x_ejemplo = [1 if sintoma in ejemplo else 0 for sintoma in todos_los_sintomas]
    
    # Crear la lista para la enfermedad
    y_ejemplo = [1 if enfermedad == ejemplo["enfermedad"] else 0 for enfermedad in etq_enfermedades]
    
    X.append(x_ejemplo)
    y.append(y_ejemplo)

# Convertir listas a arrays de numpy
X = np.array(X)
y = np.array(y)
