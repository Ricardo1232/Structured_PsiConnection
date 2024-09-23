import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import random

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

# Rutas de los archivos del modelo y el escalador
RUTA_MODELO = 'ia3/model/modelo_trastornos_cognitivos_huber.keras'
RUTA_ESCALADOR = 'ia3/scaler/scaler_trastornos_cognitivos_huber.joblib'

def verificar_archivos(ruta_modelo, ruta_escalador):
    """
    Verifica la existencia de los archivos del modelo y el escalador.

    Parámetros:
    - ruta_modelo: Ruta al archivo del modelo.
    - ruta_escalador: Ruta al archivo del escalador.

    Lanza un FileNotFoundError si alguno de los archivos no existe.
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en '{ruta_modelo}'. Asegúrate de que el modelo esté entrenado y guardado correctamente.")
    
    if not os.path.exists(ruta_escalador):
        raise FileNotFoundError(f"El archivo del escalador no se encuentra en '{ruta_escalador}'. Asegúrate de que el escalador esté entrenado y guardado correctamente.")

def cargar_modelo_y_escalador(ruta_modelo, ruta_escalador):
    """
    Carga el modelo y el escalador desde las rutas especificadas.

    Parámetros:
    - ruta_modelo: Ruta al archivo del modelo.
    - ruta_escalador: Ruta al archivo del escalador.

    Retorna:
    - modelo: Modelo de Keras cargado.
    - scaler: Escalador cargado.
    """
    print("Verificando la existencia de los archivos...")
    verificar_archivos(ruta_modelo, ruta_escalador)
    
    print('Cargando el modelo preentrenado...')
    modelo = load_model(ruta_modelo)
    print('Modelo cargado exitosamente.')
    
    print("Cargando el escalador...")
    scaler = joblib.load(ruta_escalador)
    print("Escalador cargado exitosamente.")
    
    return modelo, scaler

def predecir_trastornos(modelo, scaler, respuestas):
    """
    Toma un vector de respuestas y predice los trastornos.

    Parámetros:
    - modelo: Modelo de Keras entrenado.
    - scaler: Escalador de datos.
    - respuestas: lista o array de 25 elementos con valores 0, 1, 2, 3, 4.

    Retorna:
    - resultados: lista de strings con los resultados de las predicciones.
    """
    # Verificar que las respuestas tengan la longitud correcta
    if len(respuestas) != NUM_PREGUNTAS:
        raise ValueError(f"Se esperaban {NUM_PREGUNTAS} respuestas, pero se recibieron {len(respuestas)}.")
    
    # Convertir a numpy array y escalar usando el escalador cargado
    vector_entrada = np.array(respuestas).reshape(1, -1)
    vector_entrada_normalizado = scaler.transform(vector_entrada)
    
    # Realizar la predicción
    probabilidades_normalizadas = modelo.predict(vector_entrada_normalizado)[0]
    probabilidades = probabilidades_normalizadas * 100  # Convertir a porcentaje

    # Convertir probabilidades a etiquetas binarias usando el umbral
    predicciones_binarias = (probabilidades_normalizadas >= UMBRAL_PROBABILIDAD).astype(int)
    
    resultados = []
    for i, nombre in enumerate(NOMBRES_TRASTORNOS):
        estado = "Presente" if predicciones_binarias[i] == 1 else "Ausente"
        resultados.append(f"{nombre}: {estado} (Probabilidad: {probabilidades[i]:.2f}%)")
    
    return resultados

def ingresar_respuestas_manual():
    """
    Permite al usuario ingresar respuestas manualmente.

    Retorna:
    - respuestas: lista de 25 enteros entre 0 y 4.
    """
    print("\nIngrese sus respuestas al cuestionario:")
    respuestas = []
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
    return respuestas

def generar_respuestas_aleatorias():
    """
    Genera un conjunto de respuestas aleatorias para el cuestionario.

    Retorna:
    - respuestas: lista de enteros con respuestas aleatorias entre 0 y 4.
    """
    return [random.choice([0, 1, 2, 3, 4]) for _ in range(NUM_PREGUNTAS)]

def mostrar_respuestas_y_predicciones(respuestas, resultados):
    """
    Muestra las respuestas y los resultados de las predicciones de manera estructurada.

    Parámetros:
    - respuestas: lista de 25 respuestas.
    - resultados: lista de resultados de predicciones.
    """
    print("\n=== Arreglo de Respuestas ===")
    for i, indices in enumerate(indices_trastornos):
        respuestas_trastorno = respuestas[indices[0]:indices[-1]+1]
        print(f"{NOMBRES_TRASTORNOS[i]}: {respuestas_trastorno}")
    
    print("\n=== Resultados de la Predicción ===")
    for resultado in resultados:
        print(resultado)
    print("-" * 50)

def generar_multiples_predicciones(modelo, scaler, n=10):
    """
    Genera múltiples predicciones utilizando respuestas aleatorias.

    Parámetros:
    - modelo: Modelo de Keras entrenado.
    - scaler: Escalador de datos.
    - n: número de predicciones a generar.
    """
    print(f"\nGenerando {n} predicciones con respuestas aleatorias...")
    for i in range(n):
        print(f"\nPredicción {i+1}:")
        respuestas = generar_respuestas_aleatorias()
        resultados = predecir_trastornos(modelo, scaler, respuestas)
        mostrar_respuestas_y_predicciones(respuestas, resultados)

def prediccion_interactiva(modelo, scaler):
    """
    Permite al usuario realizar una predicción ingresando respuestas manualmente.

    Parámetros:
    - modelo: Modelo de Keras entrenado.
    - scaler: Escalador de datos.
    """
    print("\n=== Predicción Interactiva ===")
    respuestas = ingresar_respuestas_manual()
    resultados = predecir_trastornos(modelo, scaler, respuestas)
    mostrar_respuestas_y_predicciones(respuestas, resultados)

def main():
    """
    Función principal que ejecuta el flujo de predicción.
    """
    try:
        modelo, scaler = cargar_modelo_y_escalador(RUTA_MODELO, RUTA_ESCALADOR)
    except FileNotFoundError as e:
        print(e)
        return
    
    while True:
        print("\nSeleccione una opción:")
        print("1. Realizar una predicción con respuestas aleatorias")
        print("2. Realizar una predicción ingresando respuestas manualmente")
        print("3. Salir")
        opcion = input("Ingrese el número de la opción deseada: ")
        
        if opcion == '1':
            generar_multiples_predicciones(modelo, scaler, n=1)
        elif opcion == '2':
            prediccion_interactiva(modelo, scaler)
        elif opcion == '3':
            print("Saliendo del programa. ¡Hasta luego!")
            break
        else:
            print("Opción inválida. Por favor, seleccione una opción válida.")

if __name__ == "__main__":
    main()
