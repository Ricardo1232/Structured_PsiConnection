import torch
import os

# Ruta del archivo del modelo original
ruta_modelo_original = "modelo_trastornos.pth"  # Cambia la ruta si es necesario

# Cargar el modelo
modelo = torch.load(ruta_modelo_original, map_location=torch.device('cpu'))

# Verificar si se trata de un diccionario de estado o un modelo completo
if isinstance(modelo, dict):
    print("El archivo contiene un diccionario de estado.")
    print(f"Claves del diccionario de estado: {list(modelo.keys())[:5]}")

    # Guardar una nueva versión del diccionario de estado con la versión de PyTorch
    print("Guardando el modelo con la versión de PyTorch actual...")
    modelo['torch_version'] = torch.__version__  # Agregar la versión de PyTorch actual
    nueva_ruta = "modelo_trastornos_actualizado.pth"
    torch.save(modelo, nueva_ruta)
    print(f"Modelo guardado en {nueva_ruta} con la versión de PyTorch: {modelo['torch_version']}")
else:
    print(f"El archivo contiene un modelo completo de tipo: {type(modelo)}")

    # Guardar el modelo completo nuevamente con la versión de PyTorch actual en los metadatos
    nueva_ruta = "modelo_completo_actualizado.pth"
    torch.save({'model': modelo, 'torch_version': torch.__version__}, nueva_ruta)
    print(f"Modelo completo guardado en {nueva_ruta} con la versión de PyTorch: {torch.__version__}")

# Verificar capas y parámetros
print("\n=== Información de las capas del modelo ===")
for nombre, capa in modelo.items() if isinstance(modelo, dict) else modelo.named_modules():
    print(f"Capa: {nombre}, Tipo: {type(capa)}")

# Imprimir la versión actual de PyTorch utilizada
print(f"\nVersión de PyTorch actual: {torch.__version__}")
