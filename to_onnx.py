import torch
import torch.nn as nn
import onnx
import joblib
import numpy as np

NUM_PREGUNTAS = 25

# Definir el modelo base
class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.fc1 = nn.Linear(NUM_PREGUNTAS, 375)
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

# Definir el modelo combinado con escalador (modificado para usar MinMaxScaler)
class ModeloConEscalador(nn.Module):
    def __init__(self, modelo, min_values, scale_values):
        super(ModeloConEscalador, self).__init__()
        self.modelo = modelo
        # Guardar los valores min y scale del MinMaxScaler como tensores
        self.min_values = torch.tensor(min_values, dtype=torch.float32)
        self.scale_values = torch.tensor(scale_values, dtype=torch.float32)

    def forward(self, x):
        # Aplicar el escalado de MinMaxScaler: (x - min) / scale
        x_escalado = (x - self.min_values) / self.scale_values
        return self.modelo(x_escalado)

# Cargar el modelo original y el escalador para crear el modelo combinado
modelo = MiModelo()
modelo.load_state_dict(torch.load('modelo_trastornos.pth', map_location=torch.device('cpu')))

# Cargar el MinMaxScaler y extraer los valores mínimos y scale
escalador = joblib.load('scaler_trastornos_cognitivos.joblib')
min_values = escalador.data_min_  # Valor mínimo de cada característica
scale_values = escalador.data_range_  # (max - min) de cada característica

# Crear el modelo combinado que incluye el escalado
modelo_con_escalador = ModeloConEscalador(modelo, min_values, scale_values)

# Crear un tensor de entrada de ejemplo con la forma correcta (asumiendo que el modelo espera (1, 25))
entrada_ejemplo = torch.randn(1, NUM_PREGUNTAS, requires_grad=True)

# Exportar el modelo ONNX con nombres de entrada y salida correctos
torch.onnx.export(
    modelo_con_escalador,
    entrada_ejemplo,  # Tensor de entrada de ejemplo
    'modelo_con_escalador_ajustado_minmax.onnx',  # Archivo de salida ONNX
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],  # Definir el nombre de la entrada
    output_names=['output'],  # Definir el nombre de la salida
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Ejes dinámicos para batch size
)
print("Modelo exportado exitosamente a ONNX con las entradas definidas correctamente.")
