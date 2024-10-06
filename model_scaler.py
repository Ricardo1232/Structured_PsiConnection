import torch
import torch.nn as nn
import joblib
import os
import random
import numpy as np

# Configuración global
NUM_PREGUNTAS = 25
NOMBRES_TRASTORNOS = [
    'Depresivo Mayor',
    'Trastorno de Ansiedad Generalizada (TAG)',
    'Trastorno de Ansiedad Social',
    'Trastorno por Déficit de Atención',
    'Trastorno Antisocial de la Personalidad'
]
UMBRAL_PROBABILIDAD = 0.76  # Umbral estándar para clasificación binaria

# Definir los índices de preguntas por cada trastorno
indices_trastornos = [
    range(0, 5),   # Preguntas para Trastorno Depresivo Mayor
    range(5, 10),  # Preguntas para Trastorno de Ansiedad Generalizada
    range(10, 15), # Preguntas para Trastorno de Ansiedad Social
    range(15, 20), # Preguntas para Trastorno por Déficit de Atención
    range(20, 25)  # Preguntas para Trastorno Antisocial de la Personalidad
]

# Definir la red neuronal en PyTorch
class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.fc1 = nn.Linear(25, 375)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(375, 250)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(250, 100)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(100, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.elu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.elu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Clase que integra el modelo y el escalador
class ModeloConEscalador(nn.Module):
    def __init__(self, modelo, escalador):
        super(ModeloConEscalador, self).__init__()
        self.modelo = modelo  # El modelo de PyTorch
        self.escalador = escalador  # El escalador de scikit-learn (por ejemplo, StandardScaler)

    def forward(self, x):
        # Aplicar el escalado a las entradas antes de pasar al modelo
        x_np = x.detach().cpu().numpy()  # Convertir a numpy array para usar con el escalador
        x_np_escalado = self.escalador.transform(x_np)  # Aplicar el escalador
        x_escalado = torch.tensor(x_np_escalado, dtype=torch.float32)  # Convertir de vuelta a tensor
        return self.modelo(x_escalado)  # Pasar el tensor escalado al modelo para la predicción

# Rutas de los archivos del modelo y el escalador
RUTA_MODELO = 'model/modelo_trastornos.pth'  # Ajusta esta ruta según corresponda
RUTA_ESCALADOR = 'scaler/scaler_trastornos_cognitivos.joblib'  # Ajusta esta ruta según corresponda

def cargar_modelo_y_escalador(ruta_modelo, ruta_escalador):
    """
    Carga el modelo y el escalador desde las rutas especificadas.

    Parámetros:
    - ruta_modelo: Ruta al archivo del modelo.
    - ruta_escalador: Ruta al archivo del escalador.

    Retorna:
    - modelo_con_escalador: Un modelo combinado que incluye el escalador y el modelo.
    """
    # Cargar el modelo preentrenado
    modelo = MiModelo()
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
    modelo.eval()

    # Cargar el escalador
    escalador = joblib.load(ruta_escalador)

    # Crear y retornar el modelo combinado
    modelo_con_escalador = ModeloConEscalador(modelo, escalador)
    return modelo_con_escalador

def predecir_trastornos_con_modelo(modelo_con_escalador, respuestas):
    """
    Toma un vector de respuestas y predice los trastornos usando el modelo combinado.

    Parámetros:
    - modelo_con_escalador: Modelo combinado que contiene el escalador y el modelo.
    - respuestas: lista o array de 25 elementos con valores 0, 1, 2, 3, 4.

    Retorna:
    - resultados: lista de strings con los resultados de las predicciones.
    """
    # Verificar que las respuestas tengan la longitud correcta
    if len(respuestas) != NUM_PREGUNTAS:
        raise ValueError(f"Se esperaban {NUM_PREGUNTAS} respuestas, pero se recibieron {len(respuestas)}.")
    
    # Convertir a numpy array y a tensor de PyTorch
    vector_entrada = np.array(respuestas).reshape(1, -1)
    tensor_entrada = torch.tensor(vector_entrada, dtype=torch.float32)

    # Realizar la predicción usando el modelo combinado (con escalador integrado)
    with torch.no_grad():
        probabilidades_normalizadas = modelo_con_escalador(tensor_entrada).numpy()[0]
    
    probabilidades = probabilidades_normalizadas * 100  # Convertir a porcentaje

    # Convertir probabilidades a etiquetas binarias usando el umbral
    predicciones_binarias = (probabilidades_normalizadas >= UMBRAL_PROBABILIDAD).astype(int)
    
    resultados = []
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        resultados.append(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f}%)")
    
    return resultados
def prediccion_interactiva_con_modelo(modelo_con_escalador):
    """
    Permite al usuario realizar una predicción ingresando respuestas manualmente y muestra los resultados.
    
    Parámetros:
    - modelo_con_escalador: Modelo combinado que contiene el escalador y el modelo.
    """
    print("\n=== Predicción Interactiva con Modelo Combinado ===")
    
    while True:
        # Permitir al usuario ingresar respuestas manualmente
        respuestas = []
        print("\nIngrese sus respuestas para las 25 preguntas (valores entre 0 y 4):")
        for i in range(NUM_PREGUNTAS):
            while True:
                try:
                    respuesta = int(input(f"Pregunta {i+1} (0-4): "))
                    if respuesta not in [0, 1, 2, 3, 4]:
                        raise ValueError
                    respuestas.append(respuesta)
                    break
                except ValueError:
                    print("Respuesta inválida. Por favor, ingrese un número entero entre 0 y 4.")
        
        # Realizar la predicción con el modelo combinado
        resultados = predecir_trastornos_con_modelo(modelo_con_escalador, respuestas)
        
        # Mostrar los resultados de la predicción
        print("\n=== Resultados de la Predicción ===")
        for resultado in resultados:
            print(resultado)
        
        # Preguntar al usuario si desea hacer otra predicción
        otra_prediccion = input("\n¿Desea realizar otra predicción? (s/n): ").lower()
        if otra_prediccion != 's':
            print("Saliendo de la predicción interactiva. ¡Gracias!")
            break




def main():
    """
    Función principal que ejecuta el flujo de predicción.
    """
    try:
        # Cargar el modelo combinado con el escalador integrado
        modelo_con_escalador = cargar_modelo_y_escalador(RUTA_MODELO, RUTA_ESCALADOR)
        print("Modelo combinado con escalador cargado correctamente.")
        
        # Guardar el modelo combinado en un solo archivo
        torch.save(modelo_con_escalador, 'modelo_con_escalador_integrado.pth')
        print("Modelo combinado guardado en 'modelo_con_escalador_integrado.pth'.")

        prediccion_interactiva_con_modelo(modelo_con_escalador)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
