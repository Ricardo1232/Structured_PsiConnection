import onnx
from onnx_tf.backend import prepare

# Ruta al modelo ONNX exportado previamente
ruta_modelo_onnx = 'modelo_con_escalador_ajustado_minmax.onnx'  # Asegúrate de usar el nombre correcto del archivo ONNX

# Cargar el modelo ONNX
modelo_onnx = onnx.load(ruta_modelo_onnx)
print(f"Modelo ONNX cargado desde '{ruta_modelo_onnx}'.")

# Convertir el modelo ONNX a TensorFlow
modelo_tf = prepare(modelo_onnx)
print("Modelo ONNX convertido exitosamente a TensorFlow.")

# Guardar el modelo de TensorFlow en formato SavedModel
ruta_salida_tf = 'modelo_con_escalador_tf'
modelo_tf.export_graph(ruta_salida_tf)
print(f"Modelo de TensorFlow guardado en '{ruta_salida_tf}'.")
