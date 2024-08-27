import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random



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
todos_los_sintomas = ['ansiedad', 'cambios_apetito_peso', 'cambios_bruscos_humor_actividad', 'cambios_estado_animo_comportamiento', 'comportamiento_impulsivo_agresivo', 'compulsiones', 'desorientacion_espacial_temporal', 'desprecio_normas_sociales', 'dificultad_atencion', 'dificultad_hablar_escribir', 'dificultad_palabras_conversaciones', 'dificultad_tareas_cotidianas', 'dificultades_instrucciones', 'enfado', 'episodios_depresion', 'episodios_mania', 'falta_empatia_remordimiento', 'fatiga', 'hiperactividad', 'impulsividad', 'incapacidad_relaciones_estables', 'irritabilidad', 'irritabilidad_ruido', 'lentitud_movimientos', 'manipulacion_engano', 'nauseas', 'necesidad_escapar', 'nerviosismo', 'obsesiones', 'pensamientos_suicidio', 'perdida_interes', 'perdida_memoria', 'preocupacion_excesiva', 'problemas_concentracion', 'problemas_equilibrio_coordinacion', 'problemas_sueno', 'reconocimiento_ineficacia_control', 'rigidez_muscular', 'sentimientos_tristeza', 'sonidos_desencadenantes', 'sudoracion', 'temblor_reposo', 'tension_muscular']
etq_enfermedades = [
    "trastorno_ansiedad", "depresion", "tdah", "parkinson", "alzheimer", 
    "trastorno_bipolar", "toc", "misofonia", "trastorno_antisocial"
]



# Definición del modelo
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

# Crear el conjunto de datos
def crear_vector_sintomas(symptom_list, todos_los_sintomas):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

def generar_datos_sinteticos(enfermedades, num_samples=40000):
    sintomaticos = []
    etiquetas = []
    for _ in range(num_samples):
        enfermedad = random.choice(list(enfermedades.keys()))
        sintomas_base = enfermedades[enfermedad]
        num_sintomas = random.randint(1, len(sintomas_base))
        sintomas_variados = random.sample(sintomas_base, num_sintomas)
        if random.random() > 0.7:
            otra_enfermedad = random.choice([e for e in enfermedades if e != enfermedad])
            sintomas_variados += random.sample(enfermedades[otra_enfermedad], random.randint(1, 3))
        sintomaticos.append(crear_vector_sintomas(sintomas_variados, todos_los_sintomas))
        etiquetas.append(list(enfermedades.keys()).index(enfermedad))
    return np.array(sintomaticos), np.array(etiquetas)


X, y = generar_datos_sinteticos(enfermedades, num_samples=40000)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Número de pliegues
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=43)

accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Crear y entrenar el modelo
    input_size = X.shape[1]
    output_size = len(enfermedades)
    model = DiseaseModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    batch_size = 64
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluar el modelo
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = accuracy_score(y_val_tensor.numpy(), predicted.numpy())
        accuracies.append(accuracy)

print(f'Precisión promedio en validación cruzada: {np.mean(accuracies):.4f}')