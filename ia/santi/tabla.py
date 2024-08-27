import tkinter as tk
from tkinter import messagebox
import csv

# Ruta del archivo CSV existente
ruta_csv = 'datos_entrenamiento.csv'

# Leer las columnas del CSV
with open(ruta_csv, mode='r') as archivo:
    lector_csv = csv.DictReader(archivo)
    columnas = lector_csv.fieldnames

# Separar síntomas y trastornos
sintomas = columnas[:-9]  # Asumiendo que las últimas 9 columnas son los trastornos
trastornos = columnas[-9:]

# Preparar las preguntas y respuestas
preguntas = sintomas  # Solo síntomas, no trastornos
respuestas = {pregunta: 0 for pregunta in preguntas}
indice_pregunta = 0

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Registro de Datos")
ventana.geometry("600x400")
ventana.configure(bg="#f0f0f0")

# Crear y configurar la fuente
fuente = ('Arial', 12)
fuente_boton = ('Arial', 10, 'bold')
# Función para contar síntomas y calcular trastornos automáticamente
def calcular_trastornos():
    for trastorno in trastornos:
        sintomas_trastorno = [s for s in preguntas if s in trastorno]
        count = sum([respuestas[sintoma] for sintoma in sintomas_trastorno])
        if count / len(sintomas_trastorno) >= 0.8:
            respuestas[trastorno] = 1
        else:
            respuestas[trastorno] = 0

# Función para actualizar la pregunta y los botones
def actualizar_pregunta():
    if indice_pregunta < len(preguntas):
        pregunta_actual = preguntas[indice_pregunta]
        pregunta_label.config(text=f"¿{pregunta_actual.replace('_', ' ')}?")
    else:
        # Calcular trastornos y guardar los datos
        calcular_trastornos()
        guardar_datos()
        messagebox.showinfo("Éxito", "Datos añadidos exitosamente al archivo CSV.")
        reiniciar()

def responder_si():
    respuestas[preguntas[indice_pregunta]] = 1
    avanzar_pregunta()

def responder_no():
    respuestas[preguntas[indice_pregunta]] = 0
    avanzar_pregunta()

def avanzar_pregunta():
    global indice_pregunta
    if indice_pregunta < len(preguntas) - 1:
        indice_pregunta += 1
        actualizar_pregunta()
    else:
        # Calcular trastornos y guardar los datos
        calcular_trastornos()
        guardar_datos()
        messagebox.showinfo("Éxito", "Datos añadidos exitosamente al archivo CSV.")
        reiniciar()

def regresar_pregunta():
    global indice_pregunta
    if indice_pregunta > 0:
        indice_pregunta -= 1
        actualizar_pregunta()

def reiniciar():
    global indice_pregunta, respuestas
    respuestas = {pregunta: 0 for pregunta in preguntas}
    indice_pregunta = 0
    actualizar_pregunta()

def guardar_datos():
    # Añadir los nuevos datos al CSV
    with open(ruta_csv, mode='a', newline='') as archivo:
        escritor_csv = csv.DictWriter(archivo, fieldnames=columnas)
        escritor_csv.writerow(respuestas)

# Crear y colocar los widgets
pregunta_label = tk.Label(ventana, text="", font=fuente, bg="#f0f0f0", wraplength=500)
pregunta_label.pack(pady=20)

# Botones estilizados
btn_style = {'font': fuente_boton, 'bg': '#4CAF50', 'fg': 'white', 'width': 12, 'height': 2, 'bd': 0, 'relief': 'flat'}
btn_style_no = {'font': fuente_boton, 'bg': '#f44336', 'fg': 'white', 'width': 12, 'height': 2, 'bd': 0, 'relief': 'flat'}
btn_style_regresar = {'font': fuente_boton, 'bg': '#2196F3', 'fg': 'white', 'width': 12, 'height': 2, 'bd': 0, 'relief': 'flat'}
btn_style_nuevo = {'font': fuente_boton, 'bg': '#FF9800', 'fg': 'white', 'width': 12, 'height': 2, 'bd': 0, 'relief': 'flat'}

# Crear los botones con estilo
tk.Button(ventana, text="Sí", command=responder_si, **btn_style).pack(side=tk.LEFT, padx=20, pady=20)
tk.Button(ventana, text="No", command=responder_no, **btn_style_no).pack(side=tk.LEFT, padx=20, pady=20)
tk.Button(ventana, text="Regresar", command=regresar_pregunta, **btn_style_regresar).pack(side=tk.LEFT, padx=20, pady=20)
tk.Button(ventana, text="Nuevo Registro", command=reiniciar, **btn_style_nuevo).pack(side=tk.LEFT, padx=20, pady=20)

# Iniciar la interfaz
actualizar_pregunta()
ventana.mainloop()