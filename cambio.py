from datetime import date, datetime, timedelta

# AUTENTIFICA CADA TIPO DE USUARIO 
def auth_user(bcrypt, password, encriptar, session, cur, user, email, flash):
    for clave, u in user.items():              
        print()
        #COMPARAS EMAIL, COMPARAS EL CAMPO ACTIVO, COMPARAS SI ESTA VERIFICADO
        result = cur.execute(f"SELECT * FROM {u[0]} WHERE {u[1]}=%s AND ({u[2]} = %s OR {u[3]} = %s) AND {u[2]} IS NOT NULL", [email, 1, u[4],])
        if result > 0:
            data = cur.fetchone()
            if bcrypt.checkpw(password, data[u[5]].encode('utf-8')):                                   
                session["login"] = True
                session[u[6]] = True 
                session[u[7]] = data[u[7]]
                nombre = data.get(u[8])
                name   = nombre.encode()
                name   = encriptar.decrypt(name)
                name   = name.decode()                           
                session['name']  = name
                session[u[1]] = data[u[1]]
                session['verificado'] = data[u[3]]
                session.permanent = True
                if u[0] == 'paciente':
                    session['survey'] = data.get(u[10])
                    print(f"Dentro de auth: {session['survey']}")
                flash("Inicio de Sesión exitoso", 'success')
                cur.close()
                return u[9]
            else:
                # CONTRASEÑA INVALIDA 
                return 0
                

# VALIDA EL CAMPO Y ENVIA UN MENSAJE EN CASO DE SER ERRONEO
def verify_register(campo, mensaje, flash):
    if len(campo.strip()) == 0:
        flash(mensaje, 'danger')
        return True
        
    
# VERIFICA EL CODIGO DE VERIFICACION
def verify_code(mysql, session, dicc_verify, user_code):
    try:
        with mysql.connection.cursor() as cur:
            for clave, u in dicc_verify.items():
                cur.execute(f"SELECT * FROM {u[0]} WHERE {u[1]}=%s", [user_code])
                result = cur.fetchone()
                
                if result:
                    # Si el código es correcto, actualizar el campo "verificado" a 1
                    cur.execute(f"UPDATE {u[0]} SET {u[2]} = %s, {u[3]} = %s WHERE {u[1]} = %s", (2, 1, user_code))
                    mysql.connection.commit()
                    session.clear()
                    return True
    except Exception as e:
        print(f"Error: {e}")  # Imprimir el error para depuración
    return False

# FUNCION QUE CREA EL CODIGO DE SGURIDAD
def security_code():
    from random import sample
    let = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num = "0123456789"
    
    gen = f"{let}{num}"
    lon = 8
    ran = sample(gen, lon)
    cod = "".join(ran)
    return cod

# SE RECIBE LA INFORMACION DEL NOMBRE, APELLIDOP Y APELLIDOM PARA ENCRIPTAR
def get_information_3_attributes(encriptar, request, list_campos):
    valor_encriptado = []
    for campo in list_campos:
        valor    = request.form[campo].encode('utf-8')
        valor    = encriptar.encrypt(valor)
        valor_encriptado.append(valor)
        
    return valor_encriptado

# EDITAR GENERAL
def consult_edit(request, mysql, user, nombreCC, apellidoPCC, apellidoMCC):
    id         = request.form[user[0]]
    with mysql.connection.cursor() as editar:
        editar.execute(f"UPDATE {user[1]} set {user[2]}=%s, {user[3]}=%s, {user[4]}=%s WHERE {user[0]}=%s",
                            (nombreCC, apellidoPCC, apellidoMCC, id,))
        mysql.connection.commit()
    
# ELIMINAR GENERAR PARA CUALQUIER TIPO USUARIO 
def eliminarCuenta(request, mysql, user):
    id              = request.form[user[0]]
    activo          = None
    with mysql.connection.cursor() as Eliminar:
        Eliminar.execute(f"UPDATE {user[1]} set {user[2]}=%s WHERE {user[0]}=%s",
                            (activo, id,))
        mysql.connection.commit()
    
 
# FUNCION QUE DECODIFICA LOS CAMPOS PARA ACTIALIZAR EL DICCIONARIO
def select_and_decode_atribute(datos, list_campo, encriptar):
    list_valor = []
    for campo in  list_campo:
        # SELECCIONA Y DECODIFICA EL CAMPO
        valor = datos.get(campo)
        valor = encriptar.decrypt(valor)
        valor = valor.decode()
        
        list_valor.append(valor)

    # SE AGREGA A UN DICCIONARIO
    dicc = {campo: valor for campo, valor in zip(list_campo, list_valor)}
    return dicc 
    
    
def obtener_datos(user, consult, mysql, encriptar, tipo):
    # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
    with mysql.connection.cursor() as selector:
        if tipo == 1:
            selector.execute(f"SELECT * FROM {consult[0]} WHERE {consult[1]}  IS NOT NULL")
        elif tipo == 2:
            selector.execute(f"SELECT * FROM {consult[0]} {consult[1]} INNER JOIN practicante PR ON {consult[2]} = PR.idPrac INNER JOIN paciente PA ON {consult[3]} = PA.idPaci WHERE {consult[4]}= %s",(consult[5],))
            #seletda.execute(f"SELECT * FROM citas    C  INNER JOIN practicante PR ON C.idCitaPrac = PR.idPrac  INNER JOIN paciente PA ON C.idCitaPaci = PA.idPaci WHERE idCitaPaci=  %s",(idPaci,))
            #legcEnc.execute(f"SELECT * FROM encuesta E  INNER JOIN practicante PR ON E.idEncuPrac = PR.idPrac  INNER JOIN paciente PA ON E.idEncuPaci = PA.idPaci WHERE idEncuPrac=  %s",(idPrac,))
        elif tipo == 3:
            selector.execute(f"SELECT * FROM citas C INNER JOIN practicante P ON P.idPrac = C.idCitaPrac INNER JOIN paciente PA ON PA.idPaci = C.idCitaPaci WHERE P.idPrac=%s AND activoPrac IS NOT NULL AND estatusCita=%s",(consult[0], consult[1]))
        
        user_records =  selector.fetchall()

    # SE CREA UNA LISTA
    data_list = []
    
    # CON ESTE FOR, SE VAN OBTENIENDO LOS DATOS PARA POSTERIORMENTE DECODIFICARLOS
    for record in user_records:
        
        # SE AGREGA A UN DICCIONARIO
        decrypted_data = select_and_decode_atribute(record, user, encriptar)

        # SE ACTUALIZA EL DICCIONARIO QUE MANDA LA BD
        record.update(decrypted_data)

        # SE AGREGA A UNA LISTA ANTERIORMENTE CREADA
        data_list.append(record)

    # LA LISTA LA CONVERTIMOS A TUPLE PARA PODER USARLA CON MAYOR COMODIDAD EN EL FRONT
    data_list = tuple(data_list)
        
    return user_records, data_list


def crear_evento(dire_cita, tipo_cita, fecha_hora, correo_prac, correo_paci):
    """
    Crea un diccionario con los detalles del evento para Google Calendar.

    Args:
        dire_cita (str): Dirección o lugar de la cita.
        tipo_cita (str): Tipo de cita (Presencial o Virtual).
        fecha_hora (datetime): Fecha y hora de inicio de la cita.
        correo_prac (str): Correo electrónico del practicante.
        correo_paci (str): Correo electrónico del paciente.

    Returns:
        dict: Diccionario con la estructura del evento para Google Calendar.
    """
    # Definir la duración del evento (por defecto, 1 hora)
    duracion_evento = timedelta(hours=1)
    
    # Crear el evento
    evento = {
        'summary': 'Cita - Psiconnection',
        'location': f'{dire_cita} - Tipo: {tipo_cita}',
        'description': 'Cita con un psicólogo de Psiconnection',
        'status': 'confirmed',
        'sendUpdates': 'all',
        'start': {
            'dateTime': fecha_hora.isoformat(),
            'timeZone': 'America/Mexico_City',
        },
        'end': {
            'dateTime': (fecha_hora + duracion_evento).isoformat(),
            'timeZone': 'America/Mexico_City',
        },
        'recurrence': [
            'RRULE:FREQ=DAILY;COUNT=1'
        ],
        'attendees': [
            {'email': correo_prac},
            {'email': correo_paci}
        ],
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},  # 1 día antes
                {'method': 'popup', 'minutes': 10},       # 10 minutos antes
            ],
        },
    } 
    return evento
    
       
def date_to_age(fecha_nacimiento):
    fecha_actual = date.today()
    fecha_nacimiento = datetime.strptime(fecha_nacimiento, '%Y-%m-%d').date()
    edad = fecha_actual.year - fecha_nacimiento.year

    # Ajustar si el cumpleaños de este año aún no ha ocurrido
    if (fecha_actual.month, fecha_actual.day) < (fecha_nacimiento.month, fecha_nacimiento.day):
        edad -= 1

    return edad

    

import re

def validar_campos(request, campos_validacion, etiquetas):
    errores = []
    for campo, validacion in campos_validacion.items():
        valor = request.form.get(campo)
        etiqueta = etiquetas.get(campo, campo)  # Usa la etiqueta amigable o el nombre del campo si no hay etiqueta

        if not valor:
            errores.append(f"El campo {etiqueta} es requerido.")
            continue
        
        if 'max_length' in validacion and len(valor) > validacion['max_length']:
            errores.append(f"El campo {etiqueta} no puede tener más de {validacion['max_length']} caracteres.")
        
        if 'pattern' in validacion and not re.match(validacion['pattern'], valor):
            errores.append(f"El campo {etiqueta} no cumple con el formato requerido.")
            
    return errores