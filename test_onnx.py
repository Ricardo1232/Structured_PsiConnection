import onnx
import onnxruntime as ort
import numpy as np

# Ruta del archivo ONNX exportado
ruta_modelo_onnx = 'modelo_con_escalador_ajustado_minmax.onnx'  # Cambia esta ruta según corresponda

# Función para cargar y verificar el modelo ONNX
def cargar_modelo_onnx(ruta_modelo_onnx):
    """
    Carga y verifica el modelo ONNX.
    
    Parámetros:
    - ruta_modelo_onnx: Ruta del archivo ONNX.
    
    Retorna:
    - session: Sesión de ONNX Runtime cargada.
    """
    # Cargar el modelo ONNX
    modelo_onnx = onnx.load(ruta_modelo_onnx)
    onnx.checker.check_model(modelo_onnx)
    print(f"Modelo ONNX en '{ruta_modelo_onnx}' verificado exitosamente.")

    # Configurar el entorno de ejecución de ONNX
    session = ort.InferenceSession(ruta_modelo_onnx)
    
    # Imprimir información detallada de la sesión
    print("\n=== Información de la Sesión ONNX ===")
    print(f"Cantidad de entradas: {len(session.get_inputs())}")
    print(f"Cantidad de salidas: {len(session.get_outputs())}")

    # Verificar entradas del modelo
    if len(session.get_inputs()) > 0:
        for idx, entrada in enumerate(session.get_inputs()):
            print(f"Entrada {idx}: Nombre - {entrada.name}, Forma - {entrada.shape}, Tipo - {entrada.type}")
    else:
        print("No se encontraron entradas en el modelo ONNX.")

    # Verificar salidas del modelo
    if len(session.get_outputs()) > 0:
        for idx, salida in enumerate(session.get_outputs()):
            print(f"Salida {idx}: Nombre - {salida.name}, Forma - {salida.shape}, Tipo - {salida.type}")
    else:
        print("No se encontraron salidas en el modelo ONNX.")

    return session

# Función para realizar predicciones interactivas con el modelo ONNX
def prediccion_interactiva_onnx(session):
    """
    Permite al usuario realizar una predicción ingresando respuestas manualmente usando el modelo ONNX.
    
    Parámetros:
    - session: Sesión de ONNX Runtime cargada.
    """
    print("\n=== Predicción Interactiva con Modelo ONNX ===")

    # Obtener los nombres de entrada y salida del modelo ONNX
    input_name = session.get_inputs()[0].name if len(session.get_inputs()) > 0 else None
    output_name = session.get_outputs()[0].name if len(session.get_outputs()) > 0 else None

    if not input_name or not output_name:
        print("No se pudo obtener el nombre de la entrada o salida del modelo. Verifique el modelo ONNX.")
        return

    print(f"Nombre de entrada: {input_name}")
    print(f"Nombre de salida: {output_name}")

    while True:
        # Permitir al usuario ingresar respuestas manualmente
        respuestas = []
        print("\nIngrese sus respuestas para las 25 preguntas (valores entre 0 y 4):")
        for i in range(25):
            while True:
                try:
                    respuesta = int(input(f"Pregunta {i+1} (0-4): "))
                    if respuesta not in [0, 1, 2, 3, 4]:
                        raise ValueError
                    respuestas.append(respuesta)
                    break
                except ValueError:
                    print("Respuesta inválida. Por favor, ingrese un número entero entre 0 y 4.")
        
        # Convertir las respuestas a un tensor de entrada
        vector_entrada = np.array(respuestas).reshape(1, -1).astype(np.float32)

        # Realizar la predicción usando el modelo ONNX
        try:
            resultado = session.run([output_name], {input_name: vector_entrada})[0]
            print("\n=== Resultados de la Predicción ===")
            for i, valor in enumerate(resultado[0]):
                print(f"Trastorno {i+1}: {valor:.2f}")
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            break

        # Preguntar al usuario si desea hacer otra predicción
        otra_prediccion = input("\n¿Desea realizar otra predicción? (s/n): ").lower()
        if otra_prediccion != 's':
            print("Saliendo de la predicción interactiva. ¡Gracias!")
            break

# Función principal para ejecutar la consulta interactiva
def main():
    """
    Función principal que ejecuta el flujo de predicción interactiva con el modelo ONNX.
    """
    try:
        # Cargar el modelo ONNX
        session = cargar_modelo_onnx(ruta_modelo_onnx)
        
        # Iniciar la predicción interactiva
        prediccion_interactiva_onnx(session)
    except Exception as e:
        print(f"Error al cargar el modelo ONNX: {e}")

if __name__ == "__main__":
    main()
