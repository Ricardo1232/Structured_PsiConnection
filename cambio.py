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
def verify_code(mysql, cur, session, user, user_code, flash):
    for clave, u in user.items(): 
        result = cur.execute(f"SELECT * FROM {u[0]} WHERE {u[1]}=%s", [user_code])
        if result > 0:  
            # Si el código es correcto, actualizar el campo "verificado" a 1
            cur.execute(f"UPDATE {u[0]} SET {u[2]} = %s, {u[3]} = %s WHERE {u[1]} = %s", (2, 1, user_code))   
            mysql.connection.commit()
            flash("Registro completado con éxito", 'success')
            cur.close()
            session.clear()
            return True

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
    editar     = mysql.connection.cursor()
    editar.execute(f"UPDATE {user[1]} set {user[2]}=%s, {user[3]}=%s, {user[4]}=%s WHERE {user[0]}=%s",
                        (nombreCC, apellidoPCC, apellidoMCC, id,))
    mysql.connection.commit()
    
# ELIMINAR GENERAR PARA CUALQUIER TIPO USUARIO 
def eliminarCuenta(request, mysql, user):
    id              = request.form[user[0]]
    activo          = None
    Eliminar  = mysql.connection.cursor()
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
    selector     =  mysql.connection.cursor()
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




def crear_evento(direCita, tipoCita, fecha_hora, datetime, correoPrac, correoPaci):
    return {
        'summary': 'Cita - Psiconnection',
        'location': direCita+' - Tipo: '+tipoCita,
        'description': 'Cita con un psicologo de Psiconnection',
        'status': 'confirmed',
        'sendUpdates': 'all',
        'start': {
            'dateTime': fecha_hora.isoformat(),
            'timeZone': 'America/Mexico_City',
        },
        'end': {
            'dateTime': (fecha_hora + datetime.timedelta(hours=1)).isoformat(),
            'timeZone': 'America/Mexico_City',
        },
        'recurrence': [
            'RRULE:FREQ=DAILY;COUNT=1'
        ],
        'attendees': [
            {'email': correoPrac},
            {'email': correoPaci}
        ],
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},
                {'method': 'popup', 'minutes': 10},
            ],
        },
    }
    
       
def date_to_age(fecha):
    from datetime import date, datetime
    from math import floor
    
    fechaActual     = date.today()
    fechaNacimiento = datetime.strptime(fecha, '%Y-%m-%d')
    fechaNac = fechaNacimiento.date()
    edad = fechaActual - fechaNac
    edad = edad.days
    edad = edad/365
    return floor(edad)

    
