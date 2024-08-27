
from typing                 import List, Dict


SYMPTOMS = {
    "trastorno_ansiedad": [
        "preocupacion_excesiva", "nerviosismo", "fatiga", 
        "problemas_concentracion", "irritabilidad", "tension_muscular", 
        "problemas_sueno"
    ],
    "depresion": [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio"
    ],
    "tdah": [
        "dificultad_atencion", "hiperactividad", "impulsividad", 
        "dificultades_instrucciones"
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
        "irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", 
        "necesidad_escapar", "sonidos_desencadenantes"
    ],
    "trastorno_antisocial": [
        "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", 
        "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"
    ]
}

def count_symptoms(disorder_symptoms: List[str], reported_symptoms: List[str]) -> int:
    return sum(1 for symptom in disorder_symptoms if symptom in reported_symptoms)

def calculate_percentage(count: int, total: int) -> float:
    return (count / total) * 100

def diagnose(reported_symptoms: List[str]) -> Dict[str, float]:
    diagnoses_p = {}
    diagnoses_s = {}
    for disorder, symptoms in SYMPTOMS.items():
        count = count_symptoms(symptoms, reported_symptoms)
        percentage = calculate_percentage(count, len(symptoms))
        if percentage >= 80:
            diagnoses_p[disorder] = percentage
        if percentage >= 50 and percentage < 80:
            diagnoses_s[disorder] = percentage
    return diagnoses_p, diagnoses_s


symptoms = [
    # Ejemplos variados de síntomas para pruebas
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad"],  # Ejemplo con síntomas de ansiedad
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso"],  # Ejemplo con síntomas de depresión
    ["dificultad_atencion", "hiperactividad", "impulsividad"],  # Ejemplo con síntomas de TDAH
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos"],  # Ejemplo con síntomas de Parkinson
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento"],  # Ejemplo con síntomas de Alzheimer
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],  # Ejemplo con síntomas de trastorno bipolar
    ["obsesiones", "compulsiones"],  # Ejemplo con síntomas de TOC
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],  # Ejemplo con síntomas de misofonía
    ["desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"],  # Ejemplo con síntomas de trastorno antisocial

    # Combinaciones variadas para pruebas extensas
    ["problemas_sueno", "dificultad_atencion", "cambios_bruscos_humor_actividad", "tension_muscular"],  # Ejemplo diverso
    ["perdida_memoria", "dificultad_palabras_conversaciones", "dificultades_instrucciones"],  # Combinación con Alzheimer y TDAH
    ["cambios_apetito_peso", "problemas_sueno", "irritabilidad"],  # Combinación con depresión y ansiedad
    ["fatiga", "episodios_mania", "cambios_estado_animo_comportamiento"],  # Combinación con trastorno bipolar y fatiga
    ["manipulacion_engano", "falta_empatia_remordimiento"],  # Combinación con trastorno antisocial
    ["obsesiones", "reconocimiento_ineficacia_control", "necesidad_escapar"],  # Combinación con TOC y misofonía

    # Ejemplos adicionales con más síntomas
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular"],  # Ejemplo de ansiedad con más síntomas
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "fatiga", "problemas_sueno"],  # Ejemplo de depresión con más síntomas
    ["dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"],  # Ejemplo de TDAH con todos los síntomas
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion"],  # Ejemplo de Parkinson con más síntomas
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"],  # Ejemplo de Alzheimer con todos los síntomas
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad", "fatiga"],  # Ejemplo de trastorno bipolar con síntomas adicionales
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"],  # Ejemplo de TOC con más síntomas
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "sonidos_desencadenantes"],  # Ejemplo de misofonía con todos los síntomas
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"],  # Ejemplo de trastorno antisocial con todos los síntomas
    ["dificultad_atencion", "dificultades_instrucciones", "episodios_mania", "problemas_sueno"],  # Combinación cruzada de síntomas
    ["irritabilidad", "ansiedad", "cambios_apetito_peso", "dificultad_palabras_conversaciones", "dificultad_atencion"],  # Otra combinación diversa
    ["fatiga", "enfado", "desprecio_normas_sociales", "cambios_estado_animo_comportamiento", "necesidad_escapar"],  # Combinación compleja
    
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno" ], #trastorno_ansiedad
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "fatiga", "pensamientos_suicidio"], # depresion
    ["dificultad_atencion", "hiperactividad", "impulsividad",  "dificultades_instrucciones"], # tdah
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"], #parkinson
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"], #alzheimer
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"], #trastorno_bipolar
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"], # toc
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"], #misofonia
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"], #trastorno_antisocial
    
        # Combinaciones cruzadas de síntomas
    ["dificultad_atencion", "dificultades_instrucciones", "episodios_mania", "problemas_sueno"],  # Combinación cruzada de síntomas
    ["irritabilidad", "ansiedad", "cambios_apetito_peso", "dificultad_palabras_conversaciones", "dificultad_atencion"],  # Otra combinación diversa
    ["fatiga", "enfado", "desprecio_normas_sociales", "cambios_estado_animo_comportamiento", "necesidad_escapar"],  # Combinación compleja

    # Ejemplos con múltiples enfermedades
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "dificultad_atencion"],  # Ansiedad y TDAH
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "desprecio_normas_sociales"],  # Depresión y Trastorno antisocial
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "dificultades_instrucciones"],  # Parkinson y TDAH
    ["perdida_memoria", "dificultad_palabras_conversaciones", "cambios_estado_animo_comportamiento", "irritabilidad"],  # Alzheimer y Ansiedad
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad", "necesidad_escapar"],  # Trastorno bipolar y Misofonía
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control", "cambios_bruscos_humor_actividad"],  # TOC y Trastorno bipolar
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "desprecio_normas_sociales"],  # Misofonía y Trastorno antisocial
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "cambios_apetito_peso"],  # Trastorno antisocial y Depresión
    
     # Combinaciones de síntomas para múltiples enfermedades con alta probabilidad
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular"],  # Ansiedad y Trastorno bipolar
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "fatiga", "problemas_sueno", "pensamientos_suicidio"],  # Depresión severa
    ["dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones", "problemas_sueno"],  # TDAH con síntomas añadidos
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"],  # Parkinson avanzado
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"],  # Alzheimer avanzado
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad", "fatiga", "problemas_concentracion"],  # Trastorno bipolar severo
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control", "necesidad_escapar"],  # TOC severo
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar"],  # Misofonía con síntomas adicionales
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"],  # Trastorno antisocial avanzado

    # Combinaciones complejas
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "dificultad_atencion", "episodios_mania"],  # Ansiedad y TDAH y Trastorno bipolar
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "fatiga", "problemas_sueno", "desprecio_normas_sociales", "obsesiones"],  # Depresión severa con Trastorno antisocial y TOC
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir", "dificultades_instrucciones"],  # Parkinson avanzado y TDAH
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas", "fatiga"],  # Alzheimer avanzado y Fatiga
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad", "necesidad_escapar", "problemas_sueno"],  # Trastorno bipolar con Misofonía y problemas de sueño
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control", "necesidad_escapar", "irritabilidad"],  # TOC severo con Misofonía
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "desprecio_normas_sociales", "manipulacion_engano"],  # Misofonía y Trastorno antisocial
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables", "episodios_depresion"],  # Trastorno antisocial con Trastorno bipolar
    
    # Combinaciones variadas para síntomas en el rango de 50% a 80%
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion"],  # Ansiedad moderada
    ["sentimientos_tristeza", "cambios_apetito_peso", "fatiga"],  # Depresión moderada
    ["dificultad_atencion", "hiperactividad"],  # TDAH moderado
    ["temblor_reposo", "rigidez_muscular"],  # Parkinson moderado
    ["perdida_memoria", "cambios_estado_animo_comportamiento"],  # Alzheimer moderado
    ["episodios_mania", "cambios_bruscos_humor_actividad"],  # Trastorno bipolar moderado
    ["obsesiones", "compulsiones"],  # TOC moderado
    ["irritabilidad", "ansiedad"],  # Misofonía con síntomas secundarios
    ["desprecio_normas_sociales", "manipulacion_engano"],  # Trastorno antisocial moderado

    # Combinaciones para rango medio
    ["preocupacion_excesiva", "nerviosismo", "fatiga"],  # Ansiedad con síntomas secundarios
    ["sentimientos_tristeza", "cambios_apetito_peso"],  # Depresión con síntomas secundarios
    ["dificultad_atencion", "hiperactividad", "dificultades_instrucciones"],  # TDAH con síntomas secundarios
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos"],  # Parkinson con síntomas secundarios
    ["perdida_memoria", "cambios_estado_animo_comportamiento"],  # Alzheimer con síntomas secundarios
    ["episodios_mania", "episodios_depresion"],  # Trastorno bipolar con síntomas secundarios
    ["obsesiones", "reconocimiento_ineficacia_control"],  # TOC con síntomas secundarios
    ["irritabilidad", "ansiedad", "nauseas"],  # Misofonía con síntomas secundarios
    ["desprecio_normas_sociales", "falta_empatia_remordimiento"],  # Trastorno antisocial con síntomas secundarios

    # Ejemplos adicionales variados
    ["preocupacion_excesiva", "nerviosismo", "problemas_concentracion", "cambios_apetito_peso"],  # Ansiedad con síntomas de depresión
    ["sentimientos_tristeza", "fatiga", "problemas_sueno"],  # Depresión con problemas de sueño
    ["dificultad_atencion", "hiperactividad", "impulsividad"],  # TDAH con síntomas adicionales
    ["temblor_reposo", "rigidez_muscular", "dificultad_hablar_escribir"],  # Parkinson con síntomas adicionales
    ["perdida_memoria", "dificultad_palabras_conversaciones", "problemas_sueno"],  # Alzheimer con problemas de sueño
    ["episodios_mania", "cambios_bruscos_humor_actividad"],  # Trastorno bipolar con cambios bruscos
    ["obsesiones", "compulsiones", "necesidad_escapar"],  # TOC con síntomas adicionales
    ["irritabilidad", "enfado", "ansiedad", "nauseas"],  # Misofonía con síntomas adicionales
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento"],  # Trastorno antisocial con síntomas adicionales
    
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "dificultad_dormir", "hiperacusia"],  # Ansiedad avanzada con síntomas adicionales
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "fatiga", "problemas_sueno", "sentimientos_culpa", "pensamientos_suicidio", "desesperanza"],  # Depresión severa con síntomas añadidos
    ["dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones", "estrategias_organizacion", "desorganizacion", "irritabilidad"],  # TDAH completo con síntomas adicionales
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir", "dolor_muscular", "cambios_estado_animo"],  # Parkinson avanzado con síntomas añadidos
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas", "confusion", "dificultad_concentracion"],  # Alzheimer avanzado con síntomas añadidos
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad", "fatiga", "problemas_concentracion", "irritabilidad", "cambios_apetito"],  # Trastorno bipolar severo con síntomas adicionales
    ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control", "necesidad_escapar", "miedos_irreales", "dificultad_sueno", "inquietud"],  # TOC avanzado con síntomas añadidos
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes", "hiperacusia", "sensibilidad_luz"],  # Misofonía severa con síntomas añadidos
    ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables", "mentiras_cronicas", "agresividad"],  # Trastorno antisocial avanzado con síntomas añadidos
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "dificultades_instrucciones", "cambios_estado_animo_comportamiento", "dificultad_concentracion", "problemas_sueno"],  # Combinación de Parkinson con síntomas de TDAH y ansiedad
    
    ["dificultad_respirar", "opresion_toracica", "palpitaciones", "mareos", "sudoracion_excesiva", "sensacion_asfixia", "nauseas", "fatiga"],  # Ataques de pánico avanzados
    ["poca_atencion", "impulsividad", "dificultad_organizacion", "desmotivacion", "estrategias_estudio_inadecuadas", "problemas_relaciones_sociales", "dificultad_tareas_cotidianas"],  # TDAH con síntomas adicionales
    ["paranoia", "delirios", "alucinaciones", "pensamientos_desorganizados", "aislamiento_social", "comportamiento_irracional", "dificultad_concentracion"],  # Esquizofrenia con síntomas añadidos
    ["sintomas_maniacos", "episodios_euforicos", "hiperactividad", "habla_acelerada", "dificultad_concentracion", "cambios_apetito", "impulsividad"],  # Trastorno bipolar en fase maníaca
    ["fatiga_extrema", "sueño_excesivo", "dificultad_concentracion", "falta_energia", "cambios_estado_animo", "dificultades_tareas_cotidianas", "irritabilidad"],  # Síndrome de fatiga crónica
    ["dolor_cronico", "fatiga", "problemas_sueno", "dificultad_concentracion", "trastornos_gastrointestinales", "dificultades_movimiento", "depresion"],  # Fibromialgia con síntomas adicionales
    ["irritabilidad_excesiva", "cambios_rapidos_humor", "dificultad_sueno", "dificultad_concentracion", "fatiga", "sensibilidad_dolorosa", "cambios_apetito"],  # Trastorno límite de la personalidad con síntomas añadidos
    ["preocupacion_excesiva", "hipervigilancia", "dificultad_concentracion", "sudoracion_excesiva", "fatiga", "irritabilidad", "problemas_dormir"],  # Trastorno de ansiedad generalizada avanzado
    ["perdida_interes", "anhedonia", "cambios_apetito_peso", "dificultad_sueno", "sentimientos_culpa", "fatiga_extrema", "dificultad_concentracion"],  # Depresión mayor con síntomas añadidos
    ["dificultad_recuerdo", "desorientacion", "confusion", "problemas_navegacion", "perdida_memoria_corta", "dificultades_tareas_diarias", "cambios_estado_animo"],  # Deterioro cognitivo leve con síntomas adicionales
     
    ['problemas_equilibrio_coordinacion'],
    ['tension_muscular', 'irritabilidad', 'fatiga', 'problemas_concentracion', 'nerviosismo', 'problemas_sueno', 'preocupacion_excesiva'],
    ['dificultad_tareas_cotidianas', 'dificultad_palabras_conversaciones'],
    ['temblor_reposo', 'rigidez_muscular', 'lentitud_movimientos', 'problemas_equilibrio_coordinacion', 'dificultad_hablar_escribir'],
    ['rigidez_muscular'],
    ['hiperactividad', 'impulsividad', 'dificultades_instrucciones'],
    ['episodios_mania', 'episodios_depresion', 'cambios_bruscos_humor_actividad'],
    ['desprecio_normas_sociales', 'manipulacion_engano', 'falta_empatia_remordimiento', 'comportamiento_impulsivo_agresivo', 'incapacidad_relaciones_estables'],
    ['irritabilidad', 'problemas_sueno', 'temblor_reposo', 'enfado', 'hiperactividad', 'rigidez_muscular', 'problemas_concentracion', 'dificultad_palabras_conversaciones', 'nerviosismo', 'perdida_interes', 'desprecio_normas_sociales', 'manipulacion_engano', 'pensamientos_suicidio', 'dificultad_atencion', 'episodios_mania', 'sudoracion', 'sentimientos_tristeza', 'desorientacion_espacial_temporal', 'compulsiones', 'impulsividad', 'problemas_equilibrio_coordinacion', 'necesidad_escapar', 'dificultad_tareas_cotidianas', 'sonidos_desencadenantes', 'falta_empatia_remordimiento', 'perdida_memoria', 'cambios_estado_animo_comportamiento', 'reconocimiento_ineficacia_control'],
    ['reconocimiento_ineficacia_control', 'compulsiones'],
    ['temblor_reposo', 'rigidez_muscular', 'lentitud_movimientos', 'problemas_equilibrio_coordinacion', 'dificultad_hablar_escribir'],
    ['desprecio_normas_sociales', 'fatiga', 'tension_muscular', 'enfado', 'lentitud_movimientos', 'reconocimiento_ineficacia_control', 'compulsiones', 'nerviosismo', 'desorientacion_espacial_temporal', 'dificultad_atencion', 'cambios_apetito_peso', 'obsesiones', 'falta_empatia_remordimiento', 'sentimientos_tristeza', 'incapacidad_relaciones_estables', 'pensamientos_suicidio', 'sudoracion', 'dificultad_tareas_cotidianas', 'problemas_concentracion', 'problemas_sueno', 'manipulacion_engano', 'cambios_estado_animo_comportamiento', 'episodios_mania', 'dificultades_instrucciones'],
    ['perdida_memoria', 'dificultad_palabras_conversaciones', 'desorientacion_espacial_temporal', 'cambios_estado_animo_comportamiento', 'dificultad_tareas_cotidianas'],
    ['incapacidad_relaciones_estables', 'falta_empatia_remordimiento', 'comportamiento_impulsivo_agresivo', 'desprecio_normas_sociales', 'manipulacion_engano'],
    ['dificultades_instrucciones', 'tension_muscular', 'comportamiento_impulsivo_agresivo', 'preocupacion_excesiva', 'dificultad_tareas_cotidianas', 'cambios_estado_animo_comportamiento', 'fatiga', 'sudoracion', 'lentitud_movimientos', 'incapacidad_relaciones_estables', 'perdida_interes', 'sentimientos_tristeza'],
    ['compulsiones', 'nauseas', 'ansiedad', 'problemas_equilibrio_coordinacion', 'episodios_depresion', 'preocupacion_excesiva', 'nerviosismo', 'lentitud_movimientos', 'falta_empatia_remordimiento', 'sudoracion', 'perdida_interes', 'sentimientos_tristeza', 'desprecio_normas_sociales', 'incapacidad_relaciones_estables', 'dificultad_palabras_conversaciones', 'comportamiento_impulsivo_agresivo', 'enfado', 'manipulacion_engano'],
    ['cambios_estado_animo_comportamiento', 'episodios_mania', 'enfado', 'nerviosismo', 'lentitud_movimientos', 'obsesiones', 'impulsividad', 'sudoracion', 'dificultades_instrucciones', 'perdida_memoria', 'cambios_bruscos_humor_actividad', 'desprecio_normas_sociales', 'irritabilidad', 'dificultad_hablar_escribir', 'perdida_interes', 'temblor_reposo', 'fatiga', 'preocupacion_excesiva', 'dificultad_palabras_conversaciones', 'falta_empatia_remordimiento', 'tension_muscular', 'incapacidad_relaciones_estables', 'rigidez_muscular', 'reconocimiento_ineficacia_control', 'cambios_apetito_peso', 'compulsiones', 'problemas_equilibrio_coordinacion', 'nauseas', 'manipulacion_engano', 'dificultad_tareas_cotidianas', 'desorientacion_espacial_temporal'],
    ['sudoracion', 'nauseas', 'enfado'],
    [
        "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", 
        "irritabilidad", "tension_muscular", "problemas_sueno"
    ],
    [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio"
    ],
    [
        "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"
    ],
    [
        "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", 
        "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"
    ],
   [
        "perdida_memoria", "dificultad_palabras_conversaciones", 
        "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", 
        "dificultad_tareas_cotidianas"
    ],
    [
        "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"
    ],
    [
        "obsesiones", "compulsiones", "reconocimiento_ineficacia_control"
    ],
    [
        "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", 
        "necesidad_escapar", "sonidos_desencadenantes"
    ],
    [
        "desprecio_normas_sociales", "manipulacion_engano", 
        "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", 
        "incapacidad_relaciones_estables"
    ],
    ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas","sudoracion","necesidad_escapar"],
    ["enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    ["enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],
        [
        "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", 
        "irritabilidad", "tension_muscular", "problemas_sueno", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio", "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones",
        "temblor_reposo",
        "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"
    ],
   [
        "perdida_memoria", "dificultad_palabras_conversaciones", 
         "cambios_estado_animo_comportamiento", 
        "dificultad_tareas_cotidianas", "episodios_mania", "cambios_bruscos_humor_actividad"
    ],
    [
         "compulsiones", "reconocimiento_ineficacia_control"
    ],
    [
        "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", 
        "necesidad_escapar", "sonidos_desencadenantes",
        "desprecio_normas_sociales", "manipulacion_engano", 
        "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", 
        "incapacidad_relaciones_estables"
    ],
]
archivo = r"ia/test/txt/comprobacion.txt"
with open(archivo, "w") as file:
    for reported_symptoms in symptoms:
        # Obtén los diagnósticos
        diagnoses_p, diagnoses_s = diagnose(reported_symptoms)

        # Escribe los resultados en el archivo
        # file.write("Reported Symptoms: {}\n".format(reported_symptoms))
        file.write("P: {}\n".format(diagnoses_p))
        file.write("S: {}\n".format(diagnoses_s))
        file.write("\n")  # Agrega una línea en blanco para separar las entradas