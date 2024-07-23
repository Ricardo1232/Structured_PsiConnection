import mysql.connector
from datetime import datetime, timedelta

# Conectar a la base de datos
conn = mysql.connector.connect(
    user='root', password='root', host='localhost', database='psiconnection')
cursor = conn.cursor()
# ID del practicante
practicante_id = 4

# Fecha de inicio
start_date = datetime(2024, 7, 22)  # Ajusta esta fecha según sea necesario
num_days = 21  # Número de días para generar horarios (3 semanas)

# Generar fechas y horarios para insertar
for day in range(num_days):
    current_date = start_date + timedelta(days=day)
    
    # Verificar si el día actual es un día de la semana (lunes a viernes)
    for hour in range(8, 21):  # Horas de 08:00 a 20:00
        # Formatear fecha y hora
        fecha = current_date.date()
        print(fecha)
        hora = f"{hour:02d}:00"  # Ajustado a 5 caracteres
        
        # Insertar en la tabla horario
        cursor.execute("""
            INSERT INTO horario (fecha, hora, permitido, practicante_id)
            VALUES (%s, %s, %s, %s)
        """, (fecha, hora, 1, practicante_id))  # Suponiendo que 'permitido' es 1 (disponible)

# Confirmar los cambios
conn.commit()

# Cerrar cursor y conexión
cursor.close()
conn.close()

print("Fechas y horarios insertados correctamente.")