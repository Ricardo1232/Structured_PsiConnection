import torch
import torch.nn as nn
import numpy as np



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
        "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"
    ],
    "trastorno_antisocial": [
        "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables",
    ]
}
todos_los_sintomas = ['ansiedad', 'cambios_apetito_peso', 'cambios_bruscos_humor_actividad', 'cambios_estado_animo_comportamiento', 'comportamiento_impulsivo_agresivo', 'compulsiones', 'desorientacion_espacial_temporal', 'desprecio_normas_sociales', 'dificultad_atencion', 'dificultad_hablar_escribir', 'dificultad_palabras_conversaciones', 'dificultad_tareas_cotidianas', 'dificultades_instrucciones', 'enfado', 'episodios_depresion', 'episodios_mania', 'falta_empatia_remordimiento', 'fatiga', 'hiperactividad', 'impulsividad', 'incapacidad_relaciones_estables', 'irritabilidad', 'irritabilidad_ruido', 'lentitud_movimientos', 'manipulacion_engano', 'nauseas', 'necesidad_escapar', 'nerviosismo', 'obsesiones', 'pensamientos_suicidio', 'perdida_interes', 'perdida_memoria', 'preocupacion_excesiva', 'problemas_concentracion', 'problemas_equilibrio_coordinacion', 'problemas_sueno', 'reconocimiento_ineficacia_control', 'rigidez_muscular', 'sentimientos_tristeza', 'sonidos_desencadenantes', 'sudoracion', 'temblor_reposo', 'tension_muscular']


class ModeloEnfermedades(nn.Module):
    def __init__(self):
        super(ModeloEnfermedades, self).__init__()
        self.fc1 = nn.Linear(43, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 9)
        self.dropout = nn.Dropout(0.5)  # Regularización dropout

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Aplicar dropout
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)  # Aplicar dropout
        x = self.fc3(x)
        return x


# Cargar el modelo entrenado
modelo = ModeloEnfermedades()
modelo_guardado =  r"ia/modelo/modelo_diagnostico_sera.pt"
modelo.load_state_dict(torch.load(modelo_guardado))
modelo.eval()

# Ejemplo de vector de síntomas para predicción
# Supongamos que tenemos síntomas que corresponden a Trastorno de Ansiedad y Depresión
# vector_sintomas = np.zeros(43)
# vector_sintomas[0:7] = 1  # Síntomas de Trastorno de Ansiedad
# vector_sintomas[7:13] = 1  # Síntomas de Depresión
def crear_vector_sintomas(symptom_list):
    # print([1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas])
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]




listas_sintomas = [
    ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"],
    [
        "preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", 
        "irritabilidad", "tension_muscular", "problemas_sueno", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio", "dificultad_atencion", 
        "hiperactividad", "impulsividad", "dificultades_instrucciones", "temblor_reposo",
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


# Iterar sobre cada lista de síntomas y hacer predicciones
modelo.eval()  # Poner el modelo en modo de evaluación
with torch.no_grad():  # Desactivar el cálculo del gradiente para la evaluación
    for symptom_list in listas_sintomas:
        # Crear el vector de síntomas para la lista actual
        vector_sintomas = torch.tensor(crear_vector_sintomas(symptom_list))
        
        # Añadir una dimensión para simular un lote de tamaño 1
        vector_sintomas = vector_sintomas.unsqueeze(0)
        
        # Convertir el tensor a tipo Float
        vector_sintomas = vector_sintomas.float()
        
        # Hacer la predicción
        prediccion = modelo(vector_sintomas)
        
        probabilidades = torch.sigmoid(prediccion).squeeze()
        
        porcentajes = (probabilidades * 100).tolist()
        
        # Imprimir la predicción
        print(f"Síntomas: {symptom_list}")
        print(f"Predicción (porcentajes): {porcentajes}")
        print("-" * 50)