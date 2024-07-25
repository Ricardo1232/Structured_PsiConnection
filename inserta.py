from datetime import datetime, timedelta

def generar_horarios(practicante_id, mysql, num_days=21):

    # Fecha de inicio
    start_date = datetime.now()

    # Generar fechas y horarios para insertar
    try:
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            # Verificar si el día actual es un día de la semana (lunes a viernes)
            if current_date.weekday() < 5:  # 0-4 representa lunes a viernes
                for hour in range(8, 21):  # Horas de 08:00 a 20:00
                    # Formatear fecha y hora
                    fecha = current_date.date()
                    hora = f"{hour:02d}:00"
                    
                    # Insertar en la tabla horario
                    cursor.execute("""
                        INSERT INTO horario (fecha, hora, permitido, practicante_id)
                        VALUES (%s, %s, %s, %s)
                    """, (fecha, hora, 1, practicante_id))  # Suponiendo que 'permitido' es 1 (disponible)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Confirmar los cambios
        conn.commit()

        # Cerrar cursor y conexión
        cursor.close()
        conn.close()
        print("Fechas y horarios insertados correctamente.")

