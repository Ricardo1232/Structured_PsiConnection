from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Definición del modelo
# class DiseaseModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DiseaseModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, output_size)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.softmax(self.fc4(x))
#         return x

class DiseaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiseaseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Cambiado a Sigmoid

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # Cambiado a Sigmoid
        return x

# Definición de los síntomas y enfermedades
enfermedades = {
    "trastorno_ansiedad": [
        "preocupacion_excesiva", "nerviosismo", "fatiga_a", "problemas_concentracion", 
        "irritabilidad", "tension_muscular", "problemas_sueno_a"
    ],
    "depresion": [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", 
        "problemas_sueno_d", "fatiga_d", "pensamientos_suicidio"
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


# Lista de todos los síntomas únicos (44 en total)
# todos_los_sintomas = sorted(set(symptom for symptoms in enfermedades.values() for symptom in symptoms))
todos_los_sintomas = [
        "preocupacion_excesiva", "nerviosismo", "fatiga_a", "fatiga_d", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno_a",  # trastorno_ansiedad
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "pensamientos_suicidio","problemas_sueno_d",  # depresion
        "dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones",  # tdah
        "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir",  # parkinson
        "perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas",  # alzheimer
        "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad",  # trastorno_bipolar
        "obsesiones", "compulsiones", "reconocimiento_ineficacia_control",  # toc
        "irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes",  # misofonia
        "desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"  # trastorno_antisocial
    ]

# ['ansiedad', 'cambios_apetito_peso', 'cambios_bruscos_humor_actividad', 'cambios_estado_animo_comportamiento', 'comportamiento_impulsivo_agresivo', 'compulsiones', 'desorientacion_espacial_temporal', 'desprecio_normas_sociales', 'dificultad_atencion', 'dificultad_hablar_escribir', 'dificultad_palabras_conversaciones', 'dificultad_tareas_cotidianas', 'dificultades_instrucciones', 'enfado', 'episodios_depresion', 'episodios_mania', 'falta_empatia_remordimiento', 'fatiga', 'hiperactividad', 'impulsividad', 'incapacidad_relaciones_estables', 'irritabilidad', 'irritabilidad_ruido', 'lentitud_movimientos', 'manipulacion_engano', 'nauseas', 'necesidad_escapar', 'nerviosismo', 'obsesiones', 'pensamientos_suicidio', 'perdida_interes', 'perdida_memoria', 'preocupacion_excesiva', 'problemas_concentracion', 'problemas_equilibrio_coordinacion', 'problemas_sueno', 'reconocimiento_ineficacia_control', 'rigidez_muscular', 'sentimientos_tristeza', 'sonidos_desencadenantes', 'sudoracion', 'temblor_reposo', 'tension_muscular']

etq_enfermedades = [
    "trastorno_ansiedad", "depresion", "tdah", "parkinson", "alzheimer", 
    "trastorno_bipolar", "toc", "misofonia", "trastorno_antisocial"
]

print(todos_los_sintomas)

print(f'Total de síntomas únicos: {len(todos_los_sintomas)}')

# Crear un vector para cada enfermedad
def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

# Datos y preprocesamiento
X = np.array([crear_vector_sintomas(symptom_list) for symptom_list in enfermedades.values()])
y = np.array([i for i in range(len(enfermedades))])

# Generar datos sintéticos
def generar_datos_sinteticos(enfermedades, num_samples=40000):
    sintomaticos = []
    etiquetas = []

    for _ in range(num_samples):
        enfermedad = random.choice(list(enfermedades.keys()))
        sintomas_base = enfermedades[enfermedad]

        num_sintomas = random.randint(1, len(sintomas_base))
        sintomas_variados = random.sample(sintomas_base, num_sintomas)
        
        if random.random() > 0.5:
            otra_enfermedad = random.choice([e for e in enfermedades if e != enfermedad])
            sintomas_variados += random.sample(enfermedades[otra_enfermedad], random.randint(1, 3))

        sintomaticos.append(crear_vector_sintomas(sintomas_variados))
        etiquetas.append(list(enfermedades.keys()).index(enfermedad))
    
    return np.array(sintomaticos), np.array(etiquetas)

# Generar datos sintéticos y combinar con los datos originales
X_sintetico, y_sintetico = generar_datos_sinteticos(enfermedades, num_samples=40000)
X = np.vstack((X, X_sintetico))
y = np.concatenate((y, y_sintetico))

# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Agregar ruido
def agregar_ruido(datos, ruido_factor=0.1):
    ruido = np.random.randn(*datos.shape) * ruido_factor
    datos_con_ruido = datos + ruido
    return np.clip(datos_con_ruido, 0, 1)

X_con_ruido = agregar_ruido(X)
X = np.vstack((X, X_con_ruido))
y = np.concatenate((y, y))

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=43)

# Crear y entrenar el modelo
input_size = len(todos_los_sintomas)
output_size = len(enfermedades)
model = DiseaseModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

batch_size = 64
epochs = 30

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Evaluación y matriz de confusión
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    _, predicted = torch.max(val_outputs, 1)
    accuracy = accuracy_score(y_val.numpy(), predicted.numpy())
    report = classification_report(y_val.numpy(), predicted.numpy(), target_names=list(enfermedades.keys()))
    
    # Matriz de confusión
    cm = confusion_matrix(y_val.numpy(), predicted.numpy())
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=etq_enfermedades, yticklabels=etq_enfermedades)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()