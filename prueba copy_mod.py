import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from random import sample, choice
from symptoms_data import symptoms_data
# Diccionario de síntomas por trastorno
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
        "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables",
    ]
}

# Listado de todos los síntomas en el mismo orden
all_symptoms = sorted(set(symptom for symptoms in SYMPTOMS.values() for symptom in symptoms))

# Creación de X e y
def create_synthetic_data(num_samples=1000):
    X = []
    y = []
    
    for _ in range(num_samples):
        disorder = choice(list(SYMPTOMS.keys()))
        symptoms = SYMPTOMS[disorder]
        symptom_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
        X.append(symptom_vector)
        y.append(disorder)
    
    return X, y

X, y = create_synthetic_data(1000)

# Codificar las etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Divide los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Convierte los datos a tensores
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define el modelo
class TrastornoModel(nn.Module):
    def __init__(self):
        super(TrastornoModel, self).__init__()
        self.fc1 = nn.Linear(len(all_symptoms), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(SYMPTOMS))  # Salida de neuronas según el número de trastornos

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instancia el modelo, define la pérdida y el optimizador
model = TrastornoModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrena el modelo
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Prueba del modelo con datos de prueba

def print_summary(y_test, y_pred, label_encoder):
    total_samples = len(y_test)
    correct_predictions = sum(y_test == y_pred)
    accuracy = correct_predictions / total_samples * 100
    print(f'Total samples: {total_samples}')
    print(f'Correct predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.2f}%')
    
    print('\nDistribution of predictions:')
    unique, counts = np.unique(y_pred, return_counts=True)
    prediction_distribution = dict(zip(label_encoder.inverse_transform(unique), counts))
    for disorder, count in prediction_distribution.items():
        print(f'{disorder}: {count} predictions')

model.eval()

# Datos de prueba (Ejemplo)
test_symptoms = [
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
]
# Convertir los datos de prueba a vectores de síntomas
def preprocess_test_data(symptoms_list):
    preprocessed = []
    for symptoms in symptoms_list:
        symptom_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
        preprocessed.append(symptom_vector)
    return torch.tensor(preprocessed, dtype=torch.float32)

X_test_data = preprocess_test_data(test_symptoms)
with torch.no_grad():
    predictions = model(X_test_data)
    probabilities = torch.softmax(predictions, dim=1)  # Obtén probabilidades
    _, predicted_labels = torch.max(predictions, 1)

    # Inicializa diccionarios para almacenar los resultados
    diagnoses_p = {}
    diagnoses_s = {}

    # Recorre cada entrada
    for i, probs in enumerate(probabilities):
        print(f"Entrada {i+1}:")
        
        # Calcula las probabilidades para cada clase
        class_probabilities = probs.tolist()
        entry_diagnoses_p = {}
        entry_diagnoses_s = {}

        for j, disorder in enumerate(label_encoder.classes_):
            percentage = class_probabilities[j] * 100  # Convertir a porcentaje
            if percentage >= 80:
                entry_diagnoses_p[disorder] = percentage
            elif 50 <= percentage < 80:
                entry_diagnoses_s[disorder] = percentage

        # Si hay diagnósticos con porcentaje >= 80%
        if entry_diagnoses_p:
            print("  Diagnósticos con porcentaje >= 80%:")
            for disorder, percentage in entry_diagnoses_p.items():
                print(f"  {disorder}: {percentage:.2f}%")

        # Si hay diagnósticos con porcentaje entre 50% y 80%
        if entry_diagnoses_s:
            print("  Diagnósticos con porcentaje entre 50% y 80%:")
            for disorder, percentage in entry_diagnoses_s.items():
                print(f"  {disorder}: {percentage:.2f}%")

        # Guarda los resultados por entrada
        if entry_diagnoses_p:
            diagnoses_p[i+1] = entry_diagnoses_p
        if entry_diagnoses_s:
            diagnoses_s[i+1] = entry_diagnoses_s

# Imprimir todos los resultados con porcentaje >= 80%
print("\nDiagnósticos con porcentaje >= 80% por entrada:")
for entry, diagnoses in diagnoses_p.items():
    print(f"Entrada {entry}:")
    for disorder, percentage in diagnoses.items():
        print(f"  {disorder}: {percentage:.2f}%")

# Imprimir todos los resultados con porcentaje entre 50% y 80%
print("\nDiagnósticos con porcentaje entre 50% y 80% por entrada:")
for entry, diagnoses in diagnoses_s.items():
    print(f"Entrada {entry}:")
    for disorder, percentage in diagnoses.items():
        print(f"  {disorder}: {percentage:.2f}%")
        
        
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,  # Guarda el número de épocas si deseas reanudar el entrenamiento desde este punto
}, 'model_checkpoint.pth')