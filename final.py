import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Definir la arquitectura del modelo (debe coincidir con el modelo entrenado)
class DiseaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x


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

# todos_los_sintomas = sorted(set(symptom for symptoms in enfermedades.values() for symptom in symptoms))
todos_los_sintomas = [
    "preocupacion_excesiva", "nerviosismo", "fatiga","problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno",
    "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "fatiga", "pensamientos_suicidio",
    "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones",
    "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir",
    "perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas",
    "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad",
    "obsesiones", "compulsiones", "reconocimiento_ineficacia_control",
    "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes",
    "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"
    ]
etq_enfermedades = [
    "trastorno_ansiedad", "depresion", "tdah", "parkinson", "alzheimer", 
    "trastorno_bipolar", "toc", "misofonia", "trastorno_antisocial"
]


# Función para crear el vector de síntomas
def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

# Cargar el modelo guardado
modelo_guardado = "modelo_diagnostico.pt"
input_size = len(todos_los_sintomas)
print(input_size)
output_size = len(etq_enfermedades)
model = DiseaseModel(input_size, output_size)
model.load_state_dict(torch.load(modelo_guardado))
model.eval()

# Función de predicción
def predict(symptoms):
    input_vector = crear_vector_sintomas(symptoms)
    input_vector = np.array([input_vector], dtype=np.float32)
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)

    # Predicción
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = output.numpy()[0]

        # Ajuste de probabilidades en función de la fracción de síntomas presentes
        predictions = {}
        for i, prob in enumerate(probabilities):
            enfermedad = etq_enfermedades[i]
            sintomas_enfermedad = enfermedades[enfermedad]
            sintomas_presentes = [symptom for symptom in symptoms if symptom in sintomas_enfermedad]
            fraccion_sintomas = len(sintomas_presentes) / len(sintomas_enfermedad)
            predictions[enfermedad] = prob * fraccion_sintomas * 100  # Convertir a porcentaje

        # Generar diagnósticos principales y secundarios
        diagnoses_p = {k: v for k, v in predictions.items() if v >= 80}
        diagnoses_s = {k: v for k, v in predictions.items() if 50 <= v < 80}

        diagnoses_p = dict(sorted(diagnoses_p.items(), key=lambda item: item[1], reverse=True))
        diagnoses_s = dict(sorted(diagnoses_s.items(), key=lambda item: item[1], reverse=True))

        return diagnoses_p, diagnoses_s
# Ejemplo de uso
if __name__ == "__main__":
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
        ]
    with open("resultado_red.txt", "w") as file:
        for symptoms_input in symptoms:
            diagnoses_p, diagnoses_s = predict(symptoms_input)
            print("\n\nDiagnósticos Principales:", diagnoses_p)
            print("Diagnósticos Secundarios:", diagnoses_s)
            
            file.write("Reported Symptoms: {}\n".format(symptoms_input))
            file.write("Diagnoses Principales: {}\n".format(diagnoses_p))
            file.write("Diagnoses Secundarios: {}\n".format(diagnoses_s))
            file.write("\n")  # Agrega una línea en blanco para separar las entradas
