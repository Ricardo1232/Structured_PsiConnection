import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Definir los síntomas asociados a cada enfermedad
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


# Función para crear el vector de síntomas
def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

# Función para generar datos sintéticos
def generar_datos_sinteticos(n_samples=10000):
    X = []
    y = []
    
    for enfermedad, sintomas in SYMPTOMS.items():
        for _ in range(n_samples // len(SYMPTOMS)):
            sintomas_presentes = set(sintomas)
            sintomas_ausentes = set(todos_los_sintomas) - sintomas_presentes
            sintomas_combinados = list(sintomas_presentes) + list(np.random.choice(list(sintomas_ausentes), len(sintomas_ausentes)//2, replace=False))
            X.append(crear_vector_sintomas(sintomas_combinados))
            y.append([1 if enfermedad == e else 0 for e in SYMPTOMS.keys()])
    
    for _ in range(n_samples // 10):
        X.append(crear_vector_sintomas([]))
        y.append([0] * len(SYMPTOMS))
    
    return np.array(X), np.array(y)

# Generar datos sintéticos
X, y = generar_datos_sinteticos()

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Definir el modelo de red neuronal
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
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Función de pérdida personalizada
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, input_vector):
        # Pérdida BCE
        loss = self.bce_loss(outputs, targets)
        
        # Penalización por síntomas no presentes
        for i in range(outputs.size(1)):
            sintomas_requeridos = sum(targets[:, i])
            if sintomas_requeridos > 0:
                sintomas_presentes = torch.sum(input_vector * targets[:, i].unsqueeze(1), dim=0).float()
                porcentaje_presencia = sintomas_presentes / sintomas_requeridos
                penalizacion = (1 - porcentaje_presencia) * 0.05
                loss += penalizacion.sum()
        
        return loss

# Función de entrenamiento
def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=100):
    criterion = CustomLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        modelo.train()
        optimizer.zero_grad()
        outputs = modelo(X_train)
        loss = criterion(outputs, y_train, X_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            modelo.eval()
            with torch.no_grad():
                outputs_test = modelo(X_test)
                test_loss = criterion(outputs_test, y_test, X_test)
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Crear el modelo
modelo = ModeloEnfermedades()

# Entrenar el modelo
entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=100)

# Guardar el modelo
modelo_guardado = r"ia/modelo/modelo_diagnostico_sera.pt"
torch.save(modelo.state_dict(), modelo_guardado)
print(f"Modelo guardado en {modelo_guardado}")

# Función para predecir enfermedades
def predecir_enfermedades(modelo, sintomas):
    vector_sintomas = torch.tensor(crear_vector_sintomas(sintomas), dtype=torch.float32).unsqueeze(0)
    modelo.eval()
    with torch.no_grad():
        prediccion = modelo(vector_sintomas)
        predicciones = torch.sigmoid(prediccion).squeeze().numpy()
    
    resultados = {}
    for i, enfermedad in enumerate(SYMPTOMS.keys()):
        resultados[enfermedad] = max(0, min(100, 100 * predicciones[i]))
    
    return resultados

# Ejemplo de uso
sintomas = ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sonidos_desencadenantes"]
predicciones = predecir_enfermedades(modelo, sintomas)
print("Predicciones:", predicciones)