
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
    ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad"],  # Ejemplo con síntomas de ansiedad
    ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso"],  # Ejemplo con síntomas de depresión
    ["dificultad_atencion", "hiperactividad", "impulsividad"],  # Ejemplo con síntomas de TDAH
    ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos"],  # Ejemplo con síntomas de Parkinson
    ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento"],  # Ejemplo con síntomas de Alzheimer
    ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],  # Ejemplo con síntomas de trastorno bipolar
    ["obsesiones", "compulsiones"],  # Ejemplo con síntomas de TOC
    ["irritabilidad", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],  # Ejemplo con síntomas de misofonía
    ["desprecio_normas_sociales", "manipulacion_engano", "comportamiento_impulsivo_agresivo"]  # Ejemplo con síntomas de trastorno antisocial
    ]
with open("comprobacion.txt", "w") as file:
    for reported_symptoms in symptoms:
        # Obtén los diagnósticos
        diagnoses_p, diagnoses_s = diagnose(reported_symptoms)

        # Escribe los resultados en el archivo
        file.write("Reported Symptoms: {}\n".format(reported_symptoms))
        file.write("Diagnoses Principales: {}\n".format(diagnoses_p))
        file.write("Diagnoses Secundarios: {}\n".format(diagnoses_s))
        file.write("\n")  # Agrega una línea en blanco para separar las entradas