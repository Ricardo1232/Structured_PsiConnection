import torch
import numpy as np
import torch.nn as nn

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



# Función para cargar el modelo combinado que incluye el escalador
def cargar_modelo_combinado(ruta_modelo):
    """
    Carga el modelo combinado que incluye el escalador y el modelo neuronal.
    
    Parámetros:
    - ruta_modelo: Ruta al archivo del modelo combinado (.pth).

    Retorna:
    - modelo_con_escalador: Modelo combinado cargado.
    """
    print(f"Cargando el modelo combinado desde '{ruta_modelo}'...")
    modelo_con_escalador = torch.load(ruta_modelo, map_location=torch.device('cpu'))
    modelo_con_escalador.eval()  # Configurar en modo evaluación
    print("Modelo combinado cargado exitosamente.")
    return modelo_con_escalador

# Función para realizar predicciones con el modelo combinado
def predecir_trastornos_con_modelo(modelo_con_escalador, respuestas):
    """
    Toma un vector de respuestas y predice los trastornos usando el modelo combinado.

    Parámetros:
    - modelo_con_escalador: Modelo combinado que contiene el escalador y el modelo.
    - respuestas: lista o array de 25 elementos con valores 0, 1, 2, 3, 4.

    Retorna:
    - resultados: lista de strings con los resultados de las predicciones.
    """
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

# Función interactiva para predicciones con el modelo combinado
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

# Función principal para ejecutar la consulta interactiva
def main():
    """
    Función principal que ejecuta el flujo de predicción interactiva con el modelo combinado.
    """
    # Ruta al modelo combinado (ajusta esta ruta si es necesario)
    ruta_modelo_combinado = 'modelo_con_escalador_integrado.pth'

    # Cargar el modelo combinado que incluye el escalador y la red neuronal
    try:
        modelo_con_escalador = cargar_modelo_combinado(ruta_modelo_combinado)
        
        # Iniciar la predicción interactiva
        prediccion_interactiva_con_modelo(modelo_con_escalador)
    except FileNotFoundError:
        print(f"No se encontró el archivo del modelo en '{ruta_modelo_combinado}'. Asegúrate de que el modelo esté guardado correctamente.")

# Ejecutar el script si es llamado directamente
if __name__ == "__main__":
    main()
