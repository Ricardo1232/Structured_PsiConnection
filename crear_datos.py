import random

# Definición de los síntomas para cada trastorno
SYMPTOMS = {
    "trastorno_ansiedad": [
        "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno"
    ],
    "depresion": [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "fatiga", "pensamientos_suicidio"
    ],
    "tdah": [
        "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"
    ],
    "parkinson": [
        "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"
    ],
    "alzheimer": [
        "perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"
    ],
    "trastorno_bipolar": [
        "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"
    ],
    "toc": [
        "obsesiones", "compulsiones", "reconocimiento_ineficacia_control"
    ],
    "misofonia": [
        "irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"
    ],
    "trastorno_antisocial": [
        "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"
    ]
}

# Listado de todos los síntomas en el mismo orden
all_symptoms = sorted(set(symptom for symptoms in SYMPTOMS.values() for symptom in symptoms))

# Definición de probabilidades para asignar síntomas
probabilities = {
    "enfermedad": 0.40,      # Probabilidad de que se incluya una enfermedad
    "mezcladas": 0.15,       # Probabilidad de que se incluya una mezcla de síntomas
    "muchos": 0.20,          # Probabilidad de muchos síntomas
    "pocos": 0.30,           # Probabilidad de pocos síntomas
    "mezcladas_y_muchos": 0.10,  # Probabilidad de mezcla y muchos síntomas
}

def generate_symptoms_data(num_entries):
    data = []
    
    for _ in range(num_entries):
        entry_type = random.choices(
            list(probabilities.keys()), 
            list(probabilities.values())
        )[0]
        
        # Generar síntomas según el tipo seleccionado
        if entry_type == "enfermedad":
            # Seleccionar una enfermedad al azar
            disorder = random.choice(list(SYMPTOMS.keys()))
            symptoms = SYMPTOMS[disorder]
        elif entry_type == "mezcladas":
            # Mezclar síntomas de diferentes enfermedades
            num_symptoms = random.randint(1, len(all_symptoms))
            symptoms = random.sample(all_symptoms, num_symptoms)
        elif entry_type == "muchos":
            # Seleccionar muchos síntomas
            disorder = random.choice(list(SYMPTOMS.keys()))
            symptoms = random.sample(SYMPTOMS[disorder], len(SYMPTOMS[disorder]))
        elif entry_type == "pocos":
            # Seleccionar pocos síntomas
            disorder = random.choice(list(SYMPTOMS.keys()))
            symptoms = random.sample(SYMPTOMS[disorder], random.randint(1, 3))
        elif entry_type == "mezcladas_y_muchos":
            # Mezcla de síntomas y muchos síntomas
            num_symptoms = random.randint(4, len(all_symptoms))
            symptoms = random.sample(all_symptoms, num_symptoms)
        
        data.append(symptoms)
    
    return data

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        file.write("symptoms_data = [\n")
        for entry in data:
            file.write("    " + repr(entry) + ",\n")
        file.write("]\n")

if __name__ == "__main__":
    num_entries = 25000  # Número de entradas a generar
    symptoms_data = generate_symptoms_data(num_entries)
    save_to_file(symptoms_data, 'symptoms_data.py')
    print(f"Data has been saved to 'symptoms_data.py'.")