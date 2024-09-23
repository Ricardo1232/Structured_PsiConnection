import numpy as np
import itertools

# Definiciones
NUM_PREGUNTAS = 25
NUM_TRASTORNOS = 5
PREGUNTAS_POR_TRASTORNO = 5

# Nombres de las clases para referencia
clases = [
    'Trastorno Depresivo Mayor',
    'Trastorno de Ansiedad Generalizada',
    'Trastorno de Ansiedad Social',
    'Trastorno por Déficit de Atención',
    'Trastorno Antisocial de la Personalidad'
]

# Indices de las preguntas correspondientes a cada trastorno (5 preguntas por trastorno)
indices_trastornos = {
    0: [0, 1, 2, 3, 4],   # Trastorno Depresivo Mayor
    1: [5, 6, 7, 8, 9],   # Trastorno de Ansiedad Generalizada
    2: [10, 11, 12, 13, 14],  # Trastorno de Ansiedad Social
    3: [15, 16, 17, 18, 19],  # Trastorno por Déficit de Atención
    4: [20, 21, 22, 23, 24],  # Trastorno Antisocial de la Personalidad
}

# Función para generar un caso donde múltiples trastornos están presentes
def generar_caso_multiple_trastornos(trastorno_ids):
    """
    Genera un caso con múltiples trastornos presentes.
    
    Parámetros:
    - trastorno_ids: Lista de índices de trastornos (ej. [0, 4]).
    
    Retorna:
    - X: Vector de respuestas (numpy array de 25 elementos).
    - Y: Vector de etiquetas (numpy array de 5 elementos, int). 
    """
    X = np.zeros(NUM_PREGUNTAS, dtype=int)
    Y = np.zeros(NUM_TRASTORNOS, dtype=int) 
    for trastorno_id in trastorno_ids:
        for idx in indices_trastornos[trastorno_id]:
            X[idx] = 4  # Máxima puntuación (4)
        Y[trastorno_id] = 1  # Indicar que el trastorno está presente
    return X, Y

# Función para generar todas las combinaciones posibles de trastornos
def generar_todas_combinaciones_trastornos():
    """
    Genera todas las combinaciones posibles de trastornos presentes.
    
    Retorna:
    - lista_casos: Lista de tuplas con (X, Y) para cada combinación.
    """
    lista_casos = []
    
    # Generar combinaciones de 1 a 5 trastornos
    for num_trastornos in range(1, NUM_TRASTORNOS + 1):
        for combinacion in itertools.combinations(range(NUM_TRASTORNOS), num_trastornos):
            X, Y = generar_caso_multiple_trastornos(list(combinacion))
            lista_casos.append((X, Y))
    
    return lista_casos

# Ejemplo de uso:
# Generar todas las combinaciones posibles de trastornos
todos_los_casos = generar_todas_combinaciones_trastornos()

# Mostrar algunos casos generados como ejemplo
for i, (X, Y) in enumerate(todos_los_casos):  # Mostrar los primeros 5
    print(f'Caso {i + 1}:')
    print(f'X (Respuestas): {X}   Y (Etiquetas): {Y}')
    print()
