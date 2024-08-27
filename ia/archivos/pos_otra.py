import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data = {
    'sintomas': [
        ["preocupacion_excesiva", "nerviosismo", "fatiga", "problemas_concentracion", "irritabilidad", "tension_muscular", "problemas_sueno"],  # trastorno_ansiedad
        ["sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", "problemas_sueno", "fatiga", "pensamientos_suicidio"],  # depresion
        ["dificultad_atencion", "hiperactividad", "impulsividad", "dificultades_instrucciones"],  # tdah
        ["temblor_reposo", "rigidez_muscular", "lentitud_movimientos", "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"],  # parkinson
        ["perdida_memoria", "dificultad_palabras_conversaciones", "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", "dificultad_tareas_cotidianas"],  # alzheimer
        ["episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"],  # trastorno_bipolar
        ["obsesiones", "compulsiones", "reconocimiento_ineficacia_control"],  # toc
        ["irritabilidad_ruido", "enfado", "ansiedad", "nauseas", "sudoracion", "necesidad_escapar", "sonidos_desencadenantes"],  # misofonia
        ["desprecio_normas_sociales", "manipulacion_engano", "falta_empatia_remordimiento", "comportamiento_impulsivo_agresivo", "incapacidad_relaciones_estables"]  # trastorno_antisocial
    ],
    'enfermedad': [
        'trastorno_ansiedad',
        'depresion',
        'tdah',
        'parkinson',
        'alzheimer',
        'trastorno_bipolar',
        'toc',
        'misofonia',
        'trastorno_antisocial'
    ]
}

todos_los_sintomas = ['ansiedad', 'cambios_apetito_peso', 'cambios_bruscos_humor_actividad', 'cambios_estado_animo_comportamiento', 'comportamiento_impulsivo_agresivo', 'compulsiones', 'desorientacion_espacial_temporal', 'desprecio_normas_sociales', 'dificultad_atencion', 'dificultad_hablar_escribir', 'dificultad_palabras_conversaciones', 'dificultad_tareas_cotidianas', 'dificultades_instrucciones', 'enfado', 'episodios_depresion', 'episodios_mania', 'falta_empatia_remordimiento', 'fatiga', 'hiperactividad', 'impulsividad', 'incapacidad_relaciones_estables', 'irritabilidad', 'irritabilidad_ruido', 'lentitud_movimientos', 'manipulacion_engano', 'nauseas', 'necesidad_escapar', 'nerviosismo', 'obsesiones', 'pensamientos_suicidio', 'perdida_interes', 'perdida_memoria', 'preocupacion_excesiva', 'problemas_concentracion', 'problemas_equilibrio_coordinacion', 'problemas_sueno', 'reconocimiento_ineficacia_control', 'rigidez_muscular', 'sentimientos_tristeza', 'sonidos_desencadenantes', 'sudoracion', 'temblor_reposo', 'tension_muscular']

def crear_vector_sintomas(symptom_list):
    return [1 if symptom in symptom_list else 0 for symptom in todos_los_sintomas]

X = []
for sintomas in data['sintomas']:
    X.append(crear_vector_sintomas(sintomas))
X = np.array(X)

# Codificar las enfermedades
y = to_categorical(data['enfermedad'])
# Suponiendo que X_train, y_train, X_val, y_val están listos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)


model = Sequential([
    Dense(64, activation='relu', input_dim=43),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax')
])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)