import tensorflow as tf
import numpy as np

# Ruta al modelo de TensorFlow exportado
ruta_modelo_tf = 'modelo_con_escalador_tf'

# Función para cargar y verificar el modelo de TensorFlow
def cargar_modelo_tf(ruta_modelo_tf):
    """
    Carga y verifica el modelo de TensorFlow.
    
    Parámetros:
    - ruta_modelo_tf: Ruta al modelo de TensorFlow.
    
    Retorna:
    - modelo_tf: El modelo de TensorFlow cargado.
    """
    print(f"Cargando el modelo de TensorFlow desde '{ruta_modelo_tf}'...")
    modelo_tf = tf.saved_model.load(ruta_modelo_tf)
    print("Modelo de TensorFlow cargado exitosamente.")
    return modelo_tf

# Función para realizar predicciones interactivas con el modelo de TensorFlow
def prediccion_interactiva_tf(modelo_tf):
    """
    Permite al usuario realizar una predicción ingresando respuestas manualmente usando el modelo de TensorFlow.
    
    Parámetros:
    - modelo_tf: El modelo de TensorFlow cargado.
    """
    print("\n=== Predicción Interactiva con Modelo de TensorFlow ===")

    # Obtener la firma predeterminada del modelo de TensorFlow
    infer = modelo_tf.signatures['serving_default']

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

        # Convertir las respuestas a un tensor de entrada para TensorFlow
        vector_entrada = np.array(respuestas).reshape(1, -1).astype(np.float32)
        
        # Realizar la predicción usando el modelo de TensorFlow
        try:
            resultado = infer(input=tf.constant(vector_entrada))['output'].numpy()[0]
            print("\n=== Resultados de la Predicción ===")
            for i, valor in enumerate(resultado):
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
    Función principal que ejecuta el flujo de predicción interactiva con el modelo de TensorFlow.
    """
    try:
        # Cargar el modelo de TensorFlow
        modelo_tf = cargar_modelo_tf(ruta_modelo_tf)
        
        # Iniciar la predicción interactiva
        prediccion_interactiva_tf(modelo_tf)
    except Exception as e:
        print(f"Error al cargar el modelo de TensorFlow: {e}")

if __name__ == "__main__":
    main()
