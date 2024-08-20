import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from random import choice

# Diccionario de síntomas por trastorno
SYMPTOMS = {
    "trastorno_ansiedad": ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno"],
    "depresion": ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "fatiga", "pensamientos_suicidio"],
    "tdah": ["dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"],
    "parkinson": ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"],
    "alzheimer": ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"],
    "trastorno_bipolar": ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],
    "toc": ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"],
    "misofonia": ["irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],
    "trastorno_antisocial": ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"]
}

def create_synthetic_data(num_samples=1000):
    """Genera datos sintéticos basados en los síntomas y trastornos definidos."""
    all_symptoms = sorted(set(symptom for symptoms in SYMPTOMS.values() for symptom in symptoms))
    X, y = [], []
    
    for _ in range(num_samples):
        disorder = choice(list(SYMPTOMS.keys()))
        symptoms = SYMPTOMS[disorder]
        symptom_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
        X.append(symptom_vector)
        y.append(disorder)
    
    return np.array(X), np.array(y), all_symptoms

def preprocess_data(X, y):
    """Preprocesa los datos: codifica etiquetas y divide en entrenamiento y prueba."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=44)
    
    return X_train, X_test, y_train, y_test, label_encoder

def create_model(input_dim, output_dim):
    """Crea el modelo de red neuronal."""
    class TrastornoModel(nn.Module):
        def __init__(self):
            super(TrastornoModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return TrastornoModel()

def train_model(model, X_train, y_train, num_epochs=100):
    """Entrena el modelo con los datos proporcionados."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
def evaluate_model(model, X_test, y_test, label_encoder):
    """Evalúa el modelo y muestra el reporte de clasificación."""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
    
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f'Precisión: {accuracy:.2f}')
    print(classification_report(y_test.numpy(), predicted.numpy(), target_names=label_encoder.classes_))

def prepare_test_data(symptoms_data, all_symptoms, label_encoder):
    """Prepara los datos de prueba para el modelo."""
    X_test = []
    y_test = []
    
    for symptoms in symptoms_data:
        symptom_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
        X_test.append(symptom_vector)
        
        # Asumimos que tenemos el trastorno verdadero en una lista. Esto es solo para pruebas.
        disorder = choice(list(SYMPTOMS.keys()))  # Aquí debes reemplazarlo con la etiqueta verdadera si la conoces
        y_test.append(disorder)
    
    X_test = np.array(X_test)
    y_test = label_encoder.transform(y_test)
    
    return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# Main execution
if __name__ == "__main__":
    X, y, all_symptoms = create_synthetic_data(1000)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(X, y)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = create_model(len(all_symptoms), len(SYMPTOMS))
    train_model(model, X_train_tensor, y_train_tensor, num_epochs=100)
    
    evaluate_model(model, X_test_tensor, y_test_tensor, label_encoder)
    
    from symptoms_data import symptoms_data
    
    # Datos de prueba
    symptoms_data_l = symptoms_data
    
    X_test_custom, y_test_custom = prepare_test_data(symptoms_data_l, all_symptoms, label_encoder)
    model.eval()
    with torch.no_grad():
        custom_outputs = model(X_test_custom)
        _, custom_predicted = torch.max(custom_outputs, 1)
        
    
    # Imprime las predicciones para los datos de prueba
    custom_predictions = label_encoder.inverse_transform(custom_predicted.numpy())
    for symptoms, prediction in zip(symptoms_data, custom_predictions):
        print(f"Síntomas: {symptoms} -> Predicción: {prediction}")
