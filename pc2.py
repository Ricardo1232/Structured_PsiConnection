from __future__ import print_function

from ast                    import If
from threading              import activeCount
from time                   import time
from flask                  import Flask, render_template, request, redirect, url_for, session, flash, make_response
from flask_mysqldb          import MySQL, MySQLdb
from flask_mail             import Mail, Message
from flask_bcrypt           import bcrypt,Bcrypt
from flask_login            import LoginManager, login_user, logout_user, login_required, login_manager
from flask_wtf.csrf         import CSRFProtect
from functools              import wraps
from werkzeug.utils         import secure_filename
from datetime               import date, datetime, timedelta
from cryptography.fernet    import Fernet
from typing                 import List, Dict

from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# import cryptography
from cambio import *
import base64
import pdfkit
import os
import random
import secrets
import re
import datetime
import os.path

PCapp                                   = Flask(__name__)
mysql                                   = MySQL(PCapp)
csrf=CSRFProtect()
PCapp.config['MYSQL_HOST']              = 'localhost'
PCapp.config['MYSQL_USER']              = 'root'
PCapp.config['MYSQL_PASSWORD']          = ''
PCapp.config['MYSQL_DB']                = 'psiconnection'
PCapp.config['MYSQL_CURSORCLASS']       = 'DictCursor'
PCapp.config['UPLOAD_FOLDER']           = './static/img/'
PCapp.config['UPLOAD_FOLDER_PDF']       = './static/pdf/'


PCapp.config['MAIL_SERVER']='smtp.gmail.com'
PCapp.config['MAIL_PORT'] = 465
PCapp.config['MAIL_USERNAME'] = 'psi.connection10@gmail.com'
PCapp.config['MAIL_PASSWORD'] = 'fyfslamsopbexbdc'
PCapp.config['MAIL_USE_TLS'] = False
PCapp.config['MAIL_USE_SSL'] = True
mail = Mail(PCapp)

bcryptObj = Bcrypt(PCapp)

# GOOGLE
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'login' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

def verified_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Asumiendo que el estado de verificación se guarda en la sesión del usuario
        if 'verificado' not in session or not session['verificado']:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
        return f(*args, **kwargs)
    return decorated_function

def survey_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'survey' not in session:
            return redirect(url_for('SurveyV2modcopy'))
        return f(*args, **kwargs)
    return decorated_function

#El Def auth controla si inicia sesion o es nuevo usuario

@PCapp.route('/pythonlogin/', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        encriptar = encriptado()
        action = request.form['action']
        # AUTENTIFICACION
        if action == 'login':
            email = request.form['email']
            password = request.form['password'].encode('utf-8')
            cur = mysql.connection.cursor()
            
            # DICCIONARIO PARA EL MANEJO DE DATOS DE AUTENTIFICACION
            dicc_user = {
                'paciente'      : ['paciente',     'correoPaci',  'activoPaci',  'veriPaci',  0,  'contraPaci',  'loginPaci',  'idPaci',  'nombrePaci',  'indexPacientes', 'veriSurvey'           ],
                'practicante'   : ['practicante',  'correoPrac',  'activoPrac',  'veriPrac',  1,  'contraPrac',  'loginPrac',  'idPrac',  'nombrePrac',  'indexPracticantes'        ],
                'supervisor'    : ['supervisor',   'correoSup',   'activoSup',   'veriSup',   1,  'contraSup',   'loginSup',   'idSup',   'nombreSup',   'verPracticantesSupervisor'],
                'admin'         : ['admin',        'correoAd',    'activoAd',    'veriAd',    1,  'contraAd',    'loginAdmin', 'idAd',    'nombreAd',    'indexAdministrador'       ]
            }
            # BUSCA EL USUARIO AUTENTIFICAR
            valor = auth_user(bcrypt, password, encriptar, session, cur, dicc_user,  email, flash)
            if valor:
                return redirect(url_for(valor))
            elif valor == 0:
                flash("Contraseña incorrecta", 'danger')
            else:    
                flash("Email no registrado", 'danger')
            cur.close()
                    
        #Registro
        elif action == 'register':
            #Falta Hacer HASH a toda la informacion personal del usuario
            
            dicc_mensaje = {
                'name'      : "Por favor ingrese su nombre completo",
                'apellidop' : "Por favor ingrese su Apellido Paterno",
                'apellidom' : "Por favor ingrese su Apellido Materno",
                'gender'    : "Por favor ingrese su genero",
                'fecha'     : "Por favor ingrese su edad",
                'email'     : "Por favor ingrese su correo electrónico"
            }
            
            #Validar el nombre
            name = request.form['name']
            if verify_register(name, dicc_mensaje['name'], flash):
                return redirect(url_for('auth'))
            
            apellidop = request.form['apellidop']
            #Validar el apellido paterno
            if verify_register(apellidop, dicc_mensaje['apellidop'], flash):
                return redirect(url_for('auth'))
            
            apellidom = request.form['apellidom']
            # Validar el apellido materno
            if verify_register(apellidom, dicc_mensaje['apellidom'], flash):
                return redirect(url_for('auth'))
                       
            genero = request.form['gender']
            #Validar el genero
            if verify_register(genero, dicc_mensaje['gender'], flash):
                return redirect(url_for('auth'))
            
            fechaNacPaci = request.form['fecha_nacimiento']
            #Validar la edad
            if verify_register(fechaNacPaci, dicc_mensaje['fecha'], flash):
                return redirect(url_for('auth'))
            
            email = request.form['email']
             # Validar el correo electrónico
            if verify_register(email, dicc_mensaje['email'], flash):
                return redirect(url_for('auth'))
            
            if not re.match(r'^[^\s@]+@(udg\.com\.mx|alumnos\.udg\.mx|academicos\.udg\.mx)$', email):
                # Si el correo electrónico no es válido, mostrar un mensaje de error
                flash("Por favor ingrese un correo electrónico válido con uno de los dominios permitidos", 'danger')
                return redirect(url_for('auth'))
            

            edad = date_to_age(fechaNacPaci)
            
            password = request.form['password']
            passwordCon = request.form['passwordCon']

            if password != passwordCon:
                flash("Las contraseñas no coinciden. Por favor, inténtelo de nuevo.", 'danger')
                return redirect(url_for('auth'))

            if not re.search(r'^(?=.*[A-Z])(?=.*[!@#$%^&*()_+|}{":?><,./;\'\[\]])[A-Za-z\d!@#$%^&*()_+|}{":?><,./;\'\[\]]{8,}$', password):
                flash("La contraseña debe tener al menos 8 caracteres, una mayúscula y un carácter especial (. ? { } [ ] ; , ! # $ @ % ” ’)", 'danger')
                return redirect(url_for('auth'))

            hashed_password = bcryptObj.generate_password_hash(password).decode('utf-8')

            # Verificar si el correo ya está registrado en la base de datos
            cur = mysql.connection.cursor()
            result = cur.execute("SELECT * FROM paciente WHERE correoPaci=%s AND activoPaci IS NOT NULL", [email,])
            if result > 0:
                # Si el correo ya está registrado, mostrar un mensaje de error
                flash("El correo ya está registrado", 'danger')
                cur.close()
                return redirect(url_for('auth'))
            
            
            # Generar un código de verificación aleatorio
            #verification_code = secrets.token_hex(3)
            
            # CODIGO DE SEGURIDAD
            verification_code = security_code()
            
            name = name.encode('utf-8')
            name = encriptar.encrypt(name)
            
            apellidop = apellidop.encode('utf-8')
            apellidop = encriptar.encrypt(apellidop)
            
            apellidom = apellidom.encode('utf-8')
            apellidom = encriptar.encrypt(apellidom)
                        
            cur = mysql.connection.cursor()            
            # Guardar el usuario y el código de verificación en la base de datos
            cur.execute("INSERT INTO paciente(fechaNacPaci, nombrePaci, apellidoPPaci , apellidoMPaci, correoPaci, sexoPaci, contraPaci, codVeriPaci, activoPaci, veriPaci, edadPaci) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", 
                            (fechaNacPaci, name, apellidop, apellidom, email, genero, hashed_password, verification_code, 0, 0, edad))
            mysql.connection.commit()

            
            # MANDAR CORREO CON CODIGO DE VERIRIFICACION
            idPaci                = cur.lastrowid
            print(idPaci)
            selPaci               = mysql.connection.cursor()
            selPaci.execute("SELECT nombrePaci FROM paciente WHERE idPaci=%s",(idPaci,))
            ad                  = selPaci.fetchone()
            
            
            nombre = ad.get('nombrePaci')
            name = nombre.encode()
            name = encriptar.decrypt(name)
            name = name.decode()            
            
            # Enviar el código de verificación por correo electrónico
            msg = Message('Código de verificación', sender=PCapp.config['MAIL_USERNAME'], recipients=[email])
            msg.body = render_template('layoutmail.html', name=name ,  verification_code=verification_code)
            msg.html = render_template('layoutmail.html', name=name ,  verification_code=verification_code)
            mail.send(msg)
            
            flash("Revisa tu correo electrónico para ver los pasos para completar tu registro!", 'success')
            return redirect(url_for('verify'))

    return render_template('login.html')

SYMPTOMS = {
    "trastorno_ansiedad": [
        "preocupacion_excesiva", "nerviosismo", "fatiga", 
        "problemas_concentracion", "irritabilidad", "tension_muscular", 
        "problemas_sueno"
    ],
    "depresion": [
        "sentimientos_tristeza", "perdida_interes", "cambios_apetito_peso", 
        "problemas_sueno", "fatiga", "pensamientos_suicidio"
    ],
    "tdah": [
        "dificultad_atencion", "hiperactividad", "impulsividad", 
        "dificultades_instrucciones"
    ],
    "parkinson": [
        "temblor_reposo", "rigidez_muscular", "lentitud_movimientos", 
        "problemas_equilibrio_coordinacion", "dificultad_hablar_escribir"
    ],
    "alzheimer": [
        "perdida_memoria", "dificultad_palabras_conversaciones", 
        "desorientacion_espacial_temporal", "cambios_estado_animo_comportamiento", 
        "dificultad_tareas_cotidianas"
    ],
    "trastorno_bipolar": [
        "episodios_mania", "episodios_depresion", "cambios_bruscos_humor_actividad"
    ],
    "toc": [
        "obsesiones", "compulsiones", "reconocimiento_ineficacia_control"
    ],
    "misofonia": [
        "irritabilidad", "enfado", "ansiedad", "nauseas", "sudoracion", 
        "necesidad_escapar", "sonidos_desencadenantes"
    ]
}

def count_symptoms(disorder_symptoms: List[str], reported_symptoms: List[str]) -> int:
    return sum(1 for symptom in disorder_symptoms if symptom in reported_symptoms)

def calculate_percentage(count: int, total: int) -> float:
    return (count / total) * 100

def diagnose(reported_symptoms: List[str]) -> Dict[str, float]:
    diagnoses_p = {}
    diagnoses_s = {}
    for disorder, symptoms in SYMPTOMS.items():
        count = count_symptoms(symptoms, reported_symptoms)
        percentage = calculate_percentage(count, len(symptoms))
        if percentage >= 80:
            diagnoses_p[disorder] = percentage
        if percentage >= 50 and percentage < 80:
            diagnoses_s[disorder] = percentage
    return diagnoses_p, diagnoses_s


@PCapp.route('/SurveyV2modcopy')
def survey_v2():
    return render_template('SurveyV2modcopy.html')


    
    
@PCapp.route('/results', methods=['POST'])
def results():
        reported_symptoms = [symptom for symptom, value in request.form.items() if value == 'si']
        print(f"{reported_symptoms}ESTOY VACIO??")
        diagnoses_p, diagnoses_s = diagnose(reported_symptoms)
        print(diagnoses_p)
        print(diagnoses_s)
        
        correo_paciente = session.get('correoPaci')
        print(correo_paciente)
        
        if not correo_paciente:
            flash("Error: No se pudo identificar al paciente", 'danger')
            return redirect(url_for('auth'))        
        
        # Guardar los resultados en la base de datos
        cur = mysql.connection.cursor()
        try:
            cur.execute("UPDATE paciente SET sint_pri = %s, sint_sec = %s, veriSurvey = %s WHERE correoPaci = %s",
                        (str(diagnoses_p), str(diagnoses_s), 1, correo_paciente))
            mysql.connection.commit()
        
            # Guardar los síntomas reportados en la sesión para uso futuro si es necesario
            session['reported_symptoms'] = reported_symptoms
            session['diagnoses_p'] = diagnoses_p
            session['diagnoses_s'] = diagnoses_s
            session['survey'] = 1
        
        
            flash("Encuesta completada con éxito", 'success')
        except MySQLdb.Error as e:
            flash(f"Error al guardar los resultados de la encuesta: {e}", 'danger')
            mysql.connection.rollback()
        finally:
            cur.close()      
              
        return redirect(url_for('indexPacientes'))
    
    

@PCapp.route('/verify', methods=['GET', 'POST'])
def verify():
    if 'login' in session:
        if session['verificado'] == 2:
            return redirect(url_for('home'))
        else:
            if request.method == 'POST':
                flash("Revisa tu correo electrónico para obtener tu código de verificación", 'success')
                # Obtener el código ingresado por el usuario
                user_code = request.form['code']
                
                # Verificar si el código es correcto
                cur = mysql.connection.cursor()
                # DICCIONARIO PARA EL MANEJO DE DATOS DE VERIFICACION
                dicc_verify = {
                    'paciente'      : ["paciente",    "codVeriPaci", "veriPaci", "activoPaci"],
                    'practicante'   : ["practicante", "codVeriPrac", "veriPrac", "activoPrac"],
                    'supervisor'    : ["supervisor",  "codVeriSup",  "veriSup",  "activoSup"],
                    'admin'         : ["admin",       "codVeriAd",   "veriAd",   "activoAd"]
                }
                # VERIFICA EL CODIGO DE VERIFICACION DE CADA USUARIO
                if verify_code(mysql, cur, session, dicc_verify, user_code, flash):
                    return redirect(url_for('auth'))
                
                flash("Código de verificación incorrecto", 'danger')
                cur.close()        
            return render_template('verify.html')  
    else:
        return redirect(url_for('auth'))  


def encriptado():
    selectoken      =   mysql.connection.cursor()
    selectoken.execute("SELECT clave FROM token")
    cl              =   selectoken.fetchone()

    # SE CONVIERTE DE TUPLE A DICT
    clave   =   cl.get('clave')

    # SE CONVIERTE DE DICT A STR
    clave   =   str(clave)

    # SE CONVIERTE DE STR A BYTE
    clave   = clave.encode('utf-8')

    # SE CREA LA CLASE FERNET
    cifrado = Fernet(clave)

    return cifrado


################### - Crear Cuenta administradores - ###########################
@PCapp.route('/CrearCuentaAdmin', methods=["GET", "POST"])
def crearCuentaAdmin():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginAdmin' in session:
                if request.method == 'POST':
                    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
                    encriptar = encriptado()
                    
                    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                    list_campos = ['nombreAd', 'apellidoPAd', 'apellidoMAd']
                    
                    # SE RECIBE LA INFORMACION
                    nombreAdCC, apellidoPAdCC , apellidoMAdCC = get_information_3_attributes(encriptar, request, list_campos)

                    # CONFIRMAR CORREO CON LA BD
                    correoAd        = request.form['correoAd']

                    contraAd        = request.form['contraAd']
                    
                    hashed_password = bcryptObj.generate_password_hash(contraAd).decode('utf-8')

                    # CODIGO DE SEGURIDAD 
                    codVeriAd   = security_code()
                    activoAd    = 0
                    veriAd      = 1
                    priviAd     = 1
                    
                    # Verificar si el correo ya está registrado en la base de datos
                    cur = mysql.connection.cursor()
                    result = cur.execute("SELECT * FROM admin WHERE correoAd=%s AND activoAd IS NOT NULL", [correoAd,])
                    if result > 0:
                        # Si el correo ya está registrado, mostrar un mensaje de error
                        flash("El correo ya está registrado", 'danger')
                        cur.close()
                        return redirect(url_for('verAdministrador'))

                    regAdmin = mysql.connection.cursor()
                    regAdmin.execute("INSERT INTO admin (nombreAd, apellidoPAd, apellidoMAd, correoAd, contraAd, codVeriAd, activoAd, veriAd, priviAd) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                        (nombreAdCC, apellidoPAdCC, apellidoMAdCC, correoAd, hashed_password, codVeriAd, activoAd, veriAd, priviAd))
                    mysql.connection.commit()

                    # MANDAR CORREO CON CODIGO DE VERIRIFICACION
                    idAd                = regAdmin.lastrowid
                    
                    selAd               = mysql.connection.cursor()
                    selAd.execute("SELECT nombreAd FROM admin WHERE idAd=%s",(idAd,))
                    ad                  = selAd.fetchone()
                    

                    nombr = ad.get('nombreAd')
                    nombr = nombr.encode()
                    nombr = encriptar.decrypt(nombr)
                    nombr = nombr.decode()

                    
                    # SE MANDA EL CORREO
                    msg = Message('Código de verificación', sender=PCapp.config['MAIL_USERNAME'], recipients=[correoAd])
                    msg.body = render_template('layoutmail.html', name=nombr, verification_code=codVeriAd)
                    msg.html = render_template('layoutmail.html', name=nombr, verification_code=codVeriAd)
                    mail.send(msg)
                    

                    flash("Revisa tu correo electrónico para ver los pasos para completar tu registro!", 'success')        
                    #MANDAR A UNA VENTANA PARA QUE META EL CODIGO DE VERFICIACION
                    return redirect(url_for('verAdministrador'))
                else:
                    flash("Error al crear la cuenta", 'danger')
                    return redirect(url_for('verAdministrador'))  
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))

    
#~~~~~~~~~~~~~~~~~~~ Ver Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerAdministrador', methods=['GET', 'POST'])
def verAdministrador():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginAdmin' in session:
                    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                    encriptar = encriptado()
                    
                    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                    list_consult = ['admin', 'activoAd']
                    list_campo = ['nombreAd', 'apellidoPAd', 'apellidoMAd']
                    
                    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                    ad, datosAd = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

                    return render_template('adm_adm.html', admin = ad, datosAd = datosAd, username=session['name'], email=session['correoAd'])    
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaAdmin', methods=["GET", "POST"])
def eliminarCuentaAdmin():
    
    list_consult = ['idAd', 'admin', 'activoAd']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta eliminada con exito.')
    return redirect(url_for('verAdministrador'))


#~~~~~~~~~~~~~~~~~~~ Editar Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaAdmin', methods=["GET", "POST"])
def editarCuentaAdmin():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombreAd', 'apellidoPAd', 'apellidoMAd']
    list_campos_consulta = ['idAd', 'admin', 'nombreAd', 'apellidoPAd', 'apellidoMAd']
    
    # SE RECIBE LA INFORMACION
    nombreAdCC, apellidoPAdCC, apellidoMAdCC =  get_information_3_attributes(encriptar, request, list_campos)

    consult_edit(request,mysql,list_campos_consulta, nombreAdCC, apellidoPAdCC, apellidoMAdCC)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verAdministrador'))
    

#~~~~~~~~~~~~~~~~~~~ Index Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/indexPacientes', methods=['GET', 'POST'])
def indexPacientes():
    if 'login' in session:
        if session['verificado'] == 2:
            if session['survey'] == 1:
                if 'loginPaci' in session:
                    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                    encriptar = encriptado()

                    # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
                    idPaci = session['idPaci']


                    # FALTA PROBAR ESTO
                    selecCita       =   mysql.connection.cursor()
                    selecCita.execute("SELECT * FROM citas C INNER JOIN practicante PR ON C.idCitaPrac = PR.idPrac INNER JOIN paciente PA ON C.idCitaPaci = PA.idPaci WHERE idCitaPaci=%s AND estatusCita=%s",(idPaci,1))
                    cit              =   selecCita.fetchone()

                    if cit is not None:
                        fechaCita = cit.get('fechaCita')
                        horaCita = cit.get('horaCita')
                        # Formatear la fecha como una cadena
                        fechaCita = fechaCita.strftime('%Y-%m-%d')
                        # Crear un objeto datetime arbitrario para usar como referencia
                        ref = datetime.datetime(2000, 1, 1)
                        # Sumar el timedelta al datetime para obtener otro datetime
                        horaCita = ref + horaCita
                        # Formatear la hora como una cadena
                        horaCita = horaCita.strftime('%H:%M:%S')
                        fechaCita = datetime.datetime.strptime(fechaCita, '%Y-%m-%d').date()

                        # Convertir la cadena de hora a objeto datetime
                        horaCita = datetime.datetime.strptime(horaCita, '%H:%M:%S').time()
                        # Obtener la fecha y hora actual
                        fecha_actual = datetime.datetime.now().date()
                        hora_actual = datetime.datetime.now().time()

                        # Sumar una hora a la hora de la cita
                        horaCita_fin = (datetime.datetime.combine(datetime.date.today(), horaCita) +
                                        datetime.timedelta(hours=1)).time()

                        # Combinar la fecha actual con la hora actual
                        fecha_hora_actual = datetime.datetime.combine(fecha_actual, hora_actual)

                        # Combinar la fecha de la cita con la hora de finalización de la cita
                        fecha_hora_fin_cita = datetime.datetime.combine(fechaCita, horaCita_fin)
                        # Comparar la fecha y hora actual con la fecha y hora de finalización de la cita
                        if fecha_hora_actual >= fecha_hora_fin_cita:
                            citaRealizada = 1
                            print("La cita ya ha pasado.")
                        else:
                            citaRealizada = 2
                            print("La cita aún no ha pasado.")
                    
                    else:
                        citaRealizada = 1

                    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                    #                   0     1        2               3               4          5   
                    list_consult = ['citas', 'C', 'C.idCitaPrac', 'C.idCitaPaci', 'idCitaPaci', idPaci]
                    list_campo = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
                    
                    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                    hc, datosCitas = obtener_datos(list_campo, list_consult, mysql, encriptar, 2)

                    return render_template('index_pacientes.html', hc = hc, datosCitas = datosCitas, cit = cit, citaRealizada = citaRealizada, username=session['name'], email=session['correoPaci'])
                
                else:
                    flash("No tienes permiso para acceder a esta página", 'danger')
                    return redirect(url_for('home'))
            else:
                return redirect(url_for('survey_v2'))
        else:
            flash("Verifica tu correo electrónico para poder iniciar sesión", 'danger')
            return redirect(url_for('verify'))
    else:
        flash("Inicia sesión para continuar", 'danger')
        return redirect(url_for('auth'))


# ~~~~~~~~~~~~~~~~~~~ Ver Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerPacientes', methods=['GET', 'POST'])
def verPacientes():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()
 
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['paciente', 'activoPaci']
    list_campo = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    
    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
    pac, datosPaci = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)
    
    return render_template('verPaciente.html', paci = pac, datosPaci = datosPaci)



@PCapp.route('/VerPacientesAdm', methods=['GET', 'POST'])
def verPacientesAdm():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginAdmin' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()
   
                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                list_consult = ['paciente', 'activoPaci']
                list_campo = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']

                # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                pac, datosPaci = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

                return render_template('adm_pacie.html', paci = pac, datosPaci = datosPaci, username=session['name'], email=session['correoAd'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


@PCapp.route('/CrearCita', methods=["GET", "POST"])
@PCapp.route('/CrearCita', methods=["GET", "POST"])
def crearCita():

        let = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        num = "0123456789"

        gen = f"{let}{num}"
        lon = 8
        ran = random.sample(gen, lon)
        cod = "".join(ran)
        requestCOD = cod

        # USO SESION PARA OBTENER LOS DATOS DEL PACIENTE
        idPaci      = session['idPaci']
        correoPaci  = session['correoPaci']
        
        # RECUPERAR DATOS
        idPrac      = request.form['idPrac']
        correoPrac  = request.form['correoPrac']
        tipoCita    = request.form['tipoCita']
        fechaCita   = request.form['fechaCita']
        horaCita    = request.form['horaCita']

        # HACER FORMATO ESPECIFICO FECHAS
        fecha_hora = datetime.datetime.strptime(f'{fechaCita} {horaCita}', '%Y-%m-%d %H:%M')

        if tipoCita == "Presencial":
            direCita = "Modulo X"
        else:
            direCita = "Virtual"
        
        estatusCita = 1

        if direCita == "Modulo X":

            """Shows basic usage of the Google Calendar API.
            Prints the start and name of the next 10 events on the user's calendar.
            """
            creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            try:
                service = build('calendar', 'v3', credentials=creds)
                
                event = crear_evento(direCita, tipoCita, fecha_hora, datetime, correoPrac, correoPaci)



                event = service.events().insert(calendarId='primary', body=event, sendNotifications=True).execute()
                print("Event created: %s" % (event.get('htmlLink')))
                eventoId = event["id"]

                print(eventoId)

                editarPaciente      = mysql.connection.cursor()
                editarPaciente.execute("UPDATE paciente SET citaActPaci=%s WHERE idPaci=%s",
                            (estatusCita, idPaci,))
                mysql.connection.commit()

                editarPracticante   = mysql.connection.cursor()
                editarPracticante.execute("UPDATE practicante SET estatusCitaPrac=%s WHERE idPrac=%s",
                            (estatusCita, idPrac,))
                mysql.connection.commit()

                regCita = mysql.connection.cursor()
                regCita.execute("INSERT INTO citas (tipo, correoCitaPra, correoCitaPac, direCita, fechaCita, horaCita, estatusCita, eventoId, idCitaPrac, idCitaPaci) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (tipoCita, correoPrac, correoPaci, direCita, fechaCita, horaCita, estatusCita, eventoId, idPrac, idPaci))
                mysql.connection.commit()
                


                flash('Cita editada con exito.')
                return redirect(url_for('indexPacientes'))


            except HttpError as error:
                print('An error occurred: %s' % error)
        else:

            """Shows basic usage of the Google Calendar API.
            Prints the start and name of the next 10 events on the user's calendar.
            """
            creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            try:
                service = build('calendar', 'v3', credentials=creds)

                event = crear_evento(direCita, tipoCita, fecha_hora, datetime, correoPrac, correoPaci)
              
                event['conferenceData'] = {
                        'createRequest':{
                            'requestId': requestCOD,
                            'conferenceSolutionKey': {
                                'type': 'hangoutsMeet'
                            }
                        }
                    }



                event = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1, sendNotifications=True).execute()
                print("Event created: %s" % (event.get('htmlLink')))
                eventoId = event["id"]

                print(eventoId)

                editarPaciente      = mysql.connection.cursor()
                editarPaciente.execute("UPDATE paciente SET citaActPaci=%s WHERE idPaci=%s",
                            (estatusCita, idPaci,))
                mysql.connection.commit()

                editarPracticante   = mysql.connection.cursor()
                editarPracticante.execute("UPDATE practicante SET estatusCitaPrac=%s WHERE idPrac=%s",
                            (estatusCita, idPrac,))
                mysql.connection.commit()

                regCita = mysql.connection.cursor()
                regCita.execute("INSERT INTO citas (tipo, correoCitaPra, correoCitaPac, direCita, fechaCita, horaCita, estatusCita, eventoId, idCitaPrac, idCitaPaci) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (tipoCita, correoPrac, correoPaci, direCita, fechaCita, horaCita, estatusCita, eventoId, idPrac, idPaci))
                mysql.connection.commit()
                


                flash('Cita agendada con exito.')
                return redirect(url_for('indexPacientes'))


            except HttpError as error:
                print('An error occurred: %s' % error)
        
#~~~~~~~~~~~~~~~~~~~ Contestar Encuesta ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EncuestaPaciente')
def encuestaPaciente():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginPaci' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()

                # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
                idPaci = session['idPaci']
                datosCitas      =   []

                # FALTA PROBAR ESTO
                selecCita       =   mysql.connection.cursor()
                selecCita.execute("SELECT * FROM citas C INNER JOIN practicante PR ON C.idCitaPrac = PR.idPrac INNER JOIN paciente PA ON C.idCitaPaci = PA.idPaci WHERE idCitaPaci=%s AND estatusCita=%s",(idPaci,1))
                cit              =   selecCita.fetchone()

                # SE CREA UNA LISTA CON LOS NOMBRES DE LOS CAMPOS
                list_campo = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
                
                # SE AGREGA A UN DICCIONARIO
                noCita = select_and_decode_atribute(cit, list_campo, encriptar)
                
                idCita = cit.get('idCita')
                idPrac = cit.get('idPrac')
                nombrPR = noCita.get('nombrePrac')
                apelpPR = noCita.get('apellidoPPrac')
                apelmPR = noCita.get('apellidoMPrac')

                # SE ACTUALIZA EL DICCIONARIO QUE MANDA LA BD
                cit.update(noCita)

                
                # SE AGREGA A UNA LISTA ANTERIORMENTE CREADA
                datosCitas.append(cit)
            
                # LA LISTA LA CONVERTIMOS A TUPLE PARA PODER USARLA CON MAYOR COMODIDAD EN EL FRONT
                datosCitas = tuple(datosCitas)
                print(datosCitas)

                return render_template('encuesta_paciente.html', nombrePrac = nombrPR, apellidoPPrac = apelpPR, apellidoMPrac = apelmPR, idCita = idCita, idPrac = idPrac, username=session['name'], email=session['correoPaci'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify')) 
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))



#~~~~~~~~~~~~~~~~~~~ Respuestas Encuesta ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/ContestarEncuesta', methods=["GET", "POST"])
def contestarEncuesta():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginPaci' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                #encriptar = encriptado()

                # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
                idPaci = session['idPaci']
                idPrac = request.form['idPrac']
                idCita = request.form['idCita']
                pregunta1 = request.form['calificacion-1']
                pregunta2 = request.form['calificacion-2']
                pregunta3 = request.form['calificacion-3']
                pregunta4 = request.form['calificacion-4']
                pregunta5 = request.form['calificacion-5']
                pregunta6 = request.form['calificacion-6']
                pregunta7 = request.form['calificacion-7']
                pregunta8 = request.form['calificacion-8']

                
                regEncuesta = mysql.connection.cursor()
                regEncuesta.execute("INSERT INTO encuesta (pregunta1Encu, pregunta2Encu, pregunta3Encu, pregunta4Encu, pregunta5Encu, pregunta6Encu, pregunta7Encu, pregunta8Encu, idEncuPaci, idEncuPrac) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (pregunta1, pregunta2, pregunta3, pregunta4, pregunta5, pregunta6, pregunta7, pregunta8, idPaci, idPrac,))
                mysql.connection.commit()

                idEncuesta = regEncuesta.lastrowid

                regEncuestaCita = mysql.connection.cursor()
                regEncuestaCita.execute("UPDATE citas SET idEncuestaCita=%s, estatusCita=%s WHERE idCita=%s", (idEncuesta, 4, idCita,))
                            
                mysql.connection.commit()

                flash('Encuesta contestada con exito.')
                return redirect(url_for('indexPacientes'))
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify')) 
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


# ~~~~~~~~~~~~~~~~~~~ Ver Encuestas de Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerEncuestasPracticante/<string:idPrac>', methods=['GET', 'POST'])
def verEncuestasPracticante(idPrac):
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginSup' in session:

                # USAR SESSION PARA OBTENER EL ID DE SUPERVISOR
                idSup = session['idSup']
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()
      
                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                #                   0        1        2               3              4             5   
                list_consult = ['encuesta', 'E', 'E.idEncuPrac', 'E.idEncuPaci', 'idEncuPrac', idPrac]
                list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
                
                # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                encu, datosEncu = obtener_datos(list_campo, list_consult, mysql, encriptar, 2)
                
                return render_template('encuesta_practicante.html', datosEncu = datosEncu, username=session['name'], email=session['correoSup'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


# ~~~~~~~~~~~~~~~~~~~ Ver Resultados de Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerResultadosEncuesta/<string:idEncu>', methods=['GET', 'POST'])
def verResultadosEncuesta(idEncu):
    if not 'login' in session:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))
    
    if not session['verificado'] == 2:
        flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
        return redirect(url_for('verify'))
    
    if not 'loginSup' in session:
        flash("No tienes permiso para acceder a esta página", 'danger')
        return redirect(url_for('home'))

    # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
    selecEncuesta    =   mysql.connection.cursor()
    selecEncuesta.execute("SELECT * FROM encuesta WHERE idEncu=%s",(idEncu,))
    encu              =   selecEncuesta.fetchone()

    print(encu)

    return render_template('resultados_encuestas.html', resu = encu, username=session['name'], email=session['correoSup'])


#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Paciente ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaPaciente', methods=["GET", "POST"])
def eliminarCitaPaciente():

        # USO SESION PARA OBTENER LOS DATOS DEL PACIENTE
        
        # RECUPERAR DATOS
        idCita      = request.form['idCita']
        fechaCita   = request.form['fechaCita']
        horaCita    = request.form['horaCita']
        eventoIdCita    = request.form['eventoCita']
        
        estatusCita = 2

        # Convertir la cadena de fecha a objeto datetime
        fechaCita = datetime.datetime.strptime(fechaCita, '%Y-%m-%d').date()

        # Convertir la cadena de hora a objeto datetime
        horaCita = datetime.datetime.strptime(horaCita, '%H:%M:%S').time()

        # Obtener la fecha y hora actual
        fecha_actual = datetime.datetime.now().date()
        hora_actual = datetime.datetime.now().time()

        # Combinar la fecha actual con la hora actual
        fecha_hora_actual = datetime.datetime.combine(fecha_actual, hora_actual)

        # Calcular la diferencia entre la fecha y hora de la cita y la fecha y hora actual
        diferencia = datetime.datetime.combine(fechaCita, horaCita) - fecha_hora_actual

        # Verificar si todavía hay más de 24 horas de diferencia antes de la cita
        if diferencia.total_seconds() > 24 * 3600:
            print("Todavía hay más de 24 horas antes de la cita. Puedes cancelarla.")

            """Shows basic usage of the Google Calendar API.
            Prints the start and name of the next 10 events on the user's calendar.
            """
            creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            try:
                service = build('calendar', 'v3', credentials=creds)

                service.events().delete(calendarId='primary', eventId=eventoIdCita, sendNotifications=True).execute()

                editarCita      = mysql.connection.cursor()
                editarCita.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s",
                            (estatusCita, idCita,))
                mysql.connection.commit()

                flash('Cita eliminada con exito.')
                return redirect(url_for('indexPacientes'))


            except HttpError as error:
                print('An error occurred: %s' % error)
        
        else:
            print("Ya no puedes cancelar la cita. Ya pasaron menos de 24 horas.")
            flash("No se puede cancelar la cita, faltan menos de 24 horas")
            return redirect(url_for('indexPacientes'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Supervisor ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaSupervisor', methods=["GET", "POST"])
def eliminarCitaSupervisor():

        # USO SESION PARA OBTENER LOS DATOS DEL PACIENTE
        
        # RECUPERAR DATOS
        idCita          = request.form['idCita']
        eventoIdCita    = request.form['eventoCita']
        cancelacion     = request.form['cancelacion']

        # Verificar si todavía hay más de 24 horas de diferencia antes de la cita
        if cancelacion == 'Si':
            print("Como no pa, ahi te va tu cancelacion.")

            """Shows basic usage of the Google Calendar API.
            Prints the start and name of the next 10 events on the user's calendar.
            """
            creds = None
            # The file token.json stores the user's access and refresh tokens, and is
            # created automatically when the authorization flow completes for the first
            # time.
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            try:
                service = build('calendar', 'v3', credentials=creds)

                service.events().delete(calendarId='primary', eventId=eventoIdCita, sendNotifications=True).execute()

                estatusCita = 2
                editarCita      = mysql.connection.cursor()
                editarCita.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s",
                            (estatusCita, idCita,))
                mysql.connection.commit()

                flash('Cita eliminada con exito.')
                return redirect(url_for('verPracticantesSupervisor'))


            except HttpError as error:
                print('An error occurred: %s' % error)
        
        else:
            estatusCita = 1
            editarCita      = mysql.connection.cursor()
            editarCita.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s",
                        (estatusCita, idCita,))
            mysql.connection.commit()
            print("Nel padrino, ahuevo ahora la bebes o la derramas")
            flash("No se puede cancelar la cita, faltan menos de 24 horas")
            return redirect(url_for('verPracticantesSupervisor'))


# ~~~~~~~~~~~~~~~~~~~ Crear Cita ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/AgendarCita', methods=['GET', 'POST'])
def agendarCita():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginPaci' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()

                # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
                idPaci = session['idPaci']

                # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
                selecPrac        =   mysql.connection.cursor()
                selecPrac.execute("SELECT * FROM practicante WHERE activoPrac IS NOT NULL ORDER BY RAND() LIMIT 10")
                pra              =   selecPrac.fetchall()

                # SE CREA UNA LISTA
                datosPrac = []

                # SE CREA UNA LISTA CON LOS NOMBRES DE LOS CAMPOS
                list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
                
                # CON ESTE FOR, SE VAN OBTENIENDO LOS DATOS PARA POSTERIORMENTE DECODIFICARLOS 
                for pract in pra:

                    # SE AGREGA A UN DICCIONARIO
                    noPrac = select_and_decode_atribute(pract, list_campo, encriptar)

                    # SE ACTUALIZA EL DICCIONARIO QUE MANDA LA BD
                    pract.update(noPrac)

                    # SE AGREGA A UNA LISTA ANTERIORMENTE CREADA
                    datosPrac.append(pract)
                
                # LA LISTA LA CONVERTIMOS A TUPLE PARA PODER USARLA CON MAYOR COMODIDAD EN EL FRONT
                datosPrac = tuple(datosPrac)
                print(datosPrac)

                return render_template('agenda_cita.html', datosPrac = datosPrac, username=session['name'], email=session['correoPaci'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Pracicante ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaPracticante', methods=["GET", "POST"])
def eliminarCitaPracticante():
        
    idCita      = request.form['idCita']
    fechaCita   = request.form['fechaCita']
    horaCita    = request.form['horaCita']
    # Tipos de estatus: 1 = ACTIVA | 2 = CANCELADA | 3 = PENDIENTE POR CANCELAR | 4 = TERMINADA
    estatusCita = 3

    # Convertir la cadena de fecha a objeto datetime
    fechaCita = datetime.datetime.strptime(fechaCita, '%Y-%m-%d').date()

    # Convertir la cadena de hora a objeto datetime
    horaCita = datetime.datetime.strptime(horaCita, '%H:%M:%S').time()

    # Obtener la fecha y hora actual
    fecha_actual = datetime.datetime.now().date()
    hora_actual = datetime.datetime.now().time()

    # Combinar la fecha actual con la hora actual
    fecha_hora_actual = datetime.datetime.combine(fecha_actual, hora_actual)

    # Calcular la diferencia entre la fecha y hora de la cita y la fecha y hora actual
    diferencia = datetime.datetime.combine(fechaCita, horaCita) - fecha_hora_actual

    # Verificar si todavía hay más de 2 horas de diferencia antes de la cita
    if diferencia.total_seconds() > 2 * 3600:
        editarCita      = mysql.connection.cursor()
        editarCita.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s",
                    (estatusCita, idCita,))
        mysql.connection.commit()

        flash('Cita eliminada con exito.')
        print("Todavía hay más de 2 horas antes de la cita. Puedes cancelarla.")
        return redirect(url_for('indexPracticantes'))

    else:
        print("Ya no puedes cancelar la cita, han pasado menos de 2 horas.")
        flash("No se puede cancelar la cita, faltan menos de 2 horas")
        return redirect(url_for('indexPracticantes'))


# VER PRACTICANTES SUPERVISOR
@PCapp.route('/VerPracticantesSupervisor', methods=['GET', 'POST'])
def verPracticantesSupervisor():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginSup' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()
                
                # USAR SESSION PARA OBTENER EL ID DE SUPERVISOR
                idSup = session['idSup']

                # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
                selecPrac        =   mysql.connection.cursor()
                selecPrac.execute("SELECT * FROM supervisor S INNER JOIN practicante P ON P.idSupPrac = S.idSup WHERE S.idSup=%s AND activoSup IS NOT NULL AND P.activoPrac IS NOT NULL",(idSup,))
                pra              =   selecPrac.fetchall()

                # SE CREA UNA LISTA
                datosPrac = []

                # SE CREA UNA LISTA CON LOS NOMBRES DE LOS CAMPOS
                list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombreSup', 'apellidoPSup', 'apellidoMSup']
                
                # CON ESTE FOR, SE VAN OBTENIENDO LOS DATOS PARA POSTERIORMENTE DECODIFICARLOS 
                for sup in pra:
                    
                    # SE AGREGA A UN DICCIONARIO
                    noPrac = select_and_decode_atribute(sup, list_campo, encriptar)
                    
                    # SE ACTUALIZA EL DICCIONARIO QUE MANDA LA BD
                    sup.update(noPrac)

                    # SE AGREGA A UNA LISTA ANTERIORMENTE CREADA
                    datosPrac.append(sup)
                
                # LA LISTA LA CONVERTIMOS A TUPLE PARA PODER USARLA CON MAYOR COMODIDAD EN EL FRONT
                datosPrac = tuple(datosPrac)

                # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
                selecCitas        =   mysql.connection.cursor()
                selecCitas.execute("SELECT * FROM supervisor S INNER JOIN practicante P ON P.idSupPrac = S.idSup INNER JOIN citas C ON C.idCitaPrac = P.idPrac WHERE P.idSupPrac=%s AND activoSup IS NOT NULL AND C.estatusCita = %s AND P.activoPrac IS NOT NULL",(idSup, 3))
                cita              =   selecCitas.fetchall()

                # SE CREA UNA LISTA
                datosCitas = []

                # SE CREA UNA LISTA CON LOS NOMBRES DE LOS CAMPOS
                list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombreSup', 'apellidoPSup', 'apellidoMSup']
                
                # CON ESTE FOR, SE VAN OBTENIENDO LOS DATOS PARA POSTERIORMENTE DECODIFICARLOS 
                for cit in cita:                    

                    # SE AGREGA A UN DICCIONARIO
                    noCita = select_and_decode_atribute(cit, list_campo, encriptar)

                    # SE ACTUALIZA EL DICCIONARIO QUE MANDA LA BD
                    cit.update(noCita)

                    # SE AGREGA A UNA LISTA ANTERIORMENTE CREADA
                    datosCitas.append(cit)
                
                # LA LISTA LA CONVERTIMOS A TUPLE PARA PODER USARLA CON MAYOR COMODIDAD EN EL FRONT
                datosCitas = tuple(datosCitas)


                return render_template('index_supervisor.html', pract = pra, datosPrac = datosPrac, datosCitas = datosCitas, username=session['name'], email=session['correoSup'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))


#~~~~~~~~~~~~~~~~~~~ Ver Practicantes ADMINISTRADOR ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerPracticantesAdm', methods=['GET', 'POST'])
def verPracticantesAdm():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginAdmin' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()

                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                list_consult = ['practicante', ' activoPrac']
                list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']

                # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                pra, datosPrac = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

                return render_template('adm_pract.html', pract = pra, datosPrac = datosPrac, username=session['name'], email=session['correoAd'])
            
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))
    

#~~~~~~~~~~~~~~~~~~~ Eliminar Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPracticantesAdm', methods=["GET", "POST"])
def eliminarCuentaPracticantesAdm():

    list_consult = ['idPrac', 'practicante', 'activoPrac']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesAdm'))

    #~~~~~~~~~~~~~~~~~~~ Eliminar Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPracticantesSup', methods=["GET", "POST"])
def eliminarCuentaPracticantesSup():

    list_consult = ['idPrac', 'practicante', 'activoPrac']
    eliminarCuenta(request, mysql, list_consult)
    
    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesSupervisor'))

#~~~~~~~~~~~~~~~~~~~ Editar Practicantes ADMIN ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPracticantesAdm', methods=["GET", "POST"])
def editarCuentaPracticantesAdm():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    idPrac               = request.form['idPrac']

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    list_campos_consulta = ['idPrac', 'practicante', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    
    # SE RECIBE LA INFORMACION
    nombrePracCC, apellidoPPracCC, apellidoMPracCC = get_information_3_attributes(encriptar, request, list_campos)
    
    consult_edit(request,mysql,list_campos_consulta, nombrePracCC, apellidoPPracCC, apellidoMPracCC)

    # PARA SUBIR LA FOTO
    if request.files.get('foto'):
        foto                =   request.files['foto']
        fotoActual          =   secure_filename(foto.filename)
        foto.save(os.path.join(PCapp.config['UPLOAD_FOLDER'], fotoActual))
        picture             =   mysql.connection.cursor()
        picture.execute("UPDATE practicante SET fotoPrac=%s WHERE idPrac=%s", (fotoActual, idPrac,))
        mysql.connection.commit()
        
    mysql.close()
    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesAdm'))

#~~~~~~~~~~~~~~~~~~~ Editar Practicantes Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPracticantesSup', methods=["GET", "POST"])
def editarCuentaPracticantesSup():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()

    idPrac               = request.form['idPrac']

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    list_campos_consulta = ['idPrac', 'practicante', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    
    # SE RECIBE LA INFORMACION
    nombrePracCC, apellidoPPracCC, apellidoMPracCC =  get_information_3_attributes(encriptar, request, list_campos)
    
    consult_edit(request,mysql,list_campos_consulta, nombrePracCC, apellidoPPracCC, apellidoMPracCC)

    # PARA SUBIR LA FOTO
    if request.files.get('foto'):
        foto                =   request.files['foto']
        fotoActual          =   secure_filename(foto.filename)
        foto.save(os.path.join(PCapp.config['UPLOAD_FOLDER'], fotoActual))
        picture             =   mysql.connection.cursor()
        picture.execute("UPDATE practicante SET fotoPrac=%s WHERE idPrac=%s", (fotoActual, idPrac,))
        mysql.connection.commit()

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesSupervisor'))


#~~~~~~~~~~~~~~~~~~~ Crear Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/CrearCuentaPracticantes', methods=["GET", "POST"])
def crearCuentaPracticantes():
    if request.method == 'POST':
        #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
        encriptar = encriptado()

        # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
        list_campos = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
        
        # SE RECIBE LA INFORMACION
        nombrePracCC, apellidoPPracCC , apellidoMPracCC = get_information_3_attributes(encriptar, request, list_campos)

        sexoPrac          = request.form['sexoPrac']
        fechaNacPrac      = request.form['fechaNacPrac']
        celPrac           = request.form['celPrac']
        codigoUPrac       = request.form['codigoUPrac']
        idSupPrac         = session['idSup']
        
        # CONFIRMAR CORREO CON LA BD
        correoPrac        = request.form['correoPrac']

        # CAMBIAR EL HASH DE LA CONTRA POR BCRYPT
        contraPrac        = request.form['contraPrac']
        
        hashed_password = bcryptObj.generate_password_hash(contraPrac).decode('utf-8')
        
        edad = date_to_age(fechaNacPrac)
       
        # CODIGO DE SEGURIDAD 
        codVeriPrac  = security_code()
        activoPrac   = 0
        veriPrac     = 1

        
         # Verificar si el correo ya está registrado en la base de datos
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM practicante WHERE correoPrac=%s AND activoPrac IS NOT NULL", [correoPrac,])
        
        if result > 0:
            # Si el correo ya está registrado, mostrar un mensaje de error
            flash("El correo ya está registrado", 'danger')
            cur.close()
            return redirect(url_for('verPracticantesSupervisor'))        
        
        regPracticante = mysql.connection.cursor()
        regPracticante.execute("INSERT INTO practicante (nombrePrac, apellidoPPrac, apellidoMPrac, contraPrac, sexoPrac, codVeriPrac, correoPrac, fechaNacPrac, activoPrac, veriPrac, edadPrac, celPrac, codigoUPrac, idSupPrac) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (nombrePracCC, apellidoPPracCC, apellidoMPracCC, hashed_password, sexoPrac, codVeriPrac, correoPrac, fechaNacPrac, activoPrac, veriPrac, edad, celPrac, codigoUPrac, idSupPrac,))
        mysql.connection.commit()

        # PARA SUBIR LA FOTO
        if request.files.get('foto'):
            idPrac              =   regPracticante.lastrowid
            foto                =   request.files['foto']
            fotoActual          =   secure_filename(foto.filename)
            foto.save(os.path.join(PCapp.config['UPLOAD_FOLDER'], fotoActual))
            picture             =   mysql.connection.cursor()
            picture.execute("UPDATE practicante SET fotoPrac=%s WHERE idPrac=%s", (fotoActual, idPrac,))
            mysql.connection.commit()


        # MANDAR CORREO CON CODIGO DE VERIRIFICACION
        idPrac              = regPracticante.lastrowid
        selPrac             = mysql.connection.cursor()
        selPrac.execute("SELECT * FROM practicante WHERE idPrac=%s",(idPrac,))
        pra                 = selPrac.fetchone()

        nombr = pra.get('nombrePrac')
        nombr = nombr.encode()
        nombr = encriptar.decrypt(nombr)
        nombr = nombr.decode()

        # SE MANDA EL CORREO
        msg = Message('Código de verificación', sender=PCapp.config['MAIL_USERNAME'], recipients=[correoPrac])
        msg.body = render_template('layoutmail.html', name=nombr, verification_code=codVeriPrac)
        msg.html = render_template('layoutmail.html', name=nombr, verification_code=codVeriPrac)
        mail.send(msg)       
    
        flash('Cuenta creada con exito.')

        #MANDAR A UNA VENTANA PARA QUE META EL CODIGO DE VERFICIACION
        return redirect(url_for('verPracticantesSupervisor'))
    else:
        flash('No se pudo crear la cuenta.')
        return redirect(url_for('verPracticantesSupervisor'))


# ~~~~~~~~~~~~~~~~~~~ Editar Pacientes Admin ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPacienteAdm', methods=["GET", "POST"])
def editarCuentaPacienteAdm():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    list_campos_consulta = ['idPaci', 'paciente', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    
    # SE RECIBE LA INFORMACION
    nombrePaciCC, apellidoPPaciCC , apellidoMAdCC =  get_information_3_attributes(encriptar, request, list_campos)

    consult_edit(request, mysql, list_campos_consulta, nombrePaciCC, apellidoPPaciCC, apellidoMAdCC)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPacientesAdm'))


# ~~~~~~~~~~~~~~~~~~~ Editar Pacientes Supervisor ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPacienteSup', methods=["GET", "POST"])
def editarCuentaPacienteSup():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    list_campos_consulta = ['idPaci', 'paciente', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    
    # SE RECIBE LA INFORMACION
    nombrePaciCC, apellidoPPaciCC , apellidoMAdCC =  get_information_3_attributes(encriptar, request, list_campos)
   
    consult_edit(request, mysql, list_campos_consulta, nombrePaciCC, apellidoPPaciCC, apellidoMAdCC)
    
    flash('Cuenta editada con exito.')
    return redirect(url_for('verPaciente'))

#~~~~~~~~~~~~~~~~~~~ Eliminar Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPacienteAdm', methods=["GET", "POST"])
def eliminarCuentaPacienteAdm():
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['idPaci', 'paciente', 'activoPaci']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPacientesAdm'))

#~~~~~~~~~~~~~~~~~~~ Eliminar Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPacienteSup', methods=["GET", "POST"])
def eliminarCuentaPacienteSup():
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['idPaci', 'paciente', 'activoPaci']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPaciente'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~ CRUD Supervisores ~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


#~~~~~~~~~~~~~~~~~~~ Crear Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/CrearCuentaSupervisor', methods=["GET", "POST"])
def crearCuentaSupervisor():
    if request.method == 'POST':
        # -------------------------------------------------------------------------
        #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
        encriptar = encriptado()
        
        # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
        list_campos = ['nombreSup', 'apellidoPSup', 'apellidoMSup']
        
        # SE RECIBE LA INFORMACION
        nombreSupCC, apellidoPSupCC, apellidoMSupCC = get_information_3_attributes(encriptar, request, list_campos)

        # CONFIRMAR CORREO CON LA BD
        correoSup        = request.form['correoSup']

        contraSup        = request.form['contraSup']
        hashed_password = bcryptObj.generate_password_hash(contraSup).decode('utf-8')

        # CODIGO DE SEGURIDAD
        codVeriSup  = security_code()
        activoSup   = 0
        veriSup     = 1
        priviSup    = 2
        
        # Verificar si el correo ya está registrado en la base de datos
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM supervisor WHERE correoSup=%s AND activoSup IS NOT NULL", [correoSup,])
        if result > 0:
            # Si el correo ya está registrado, mostrar un mensaje de error
            flash("El correo ya está registrado", 'danger')
            cur.close()
            return redirect(url_for('verSupervisor')) 

        regSupervisor = mysql.connection.cursor()
        regSupervisor.execute("INSERT INTO supervisor (nombreSup, apellidoPSup, apellidoMSup, correoSup, contraSup, codVeriSup, activoSup, veriSup, priviSup) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (nombreSupCC, apellidoPSupCC, apellidoMSupCC, correoSup, hashed_password, codVeriSup, activoSup, veriSup, priviSup))
        mysql.connection.commit()

        # MANDAR CORREO CON CODIGO DE VERIRIFICACION

        idSup               = regSupervisor.lastrowid
        selSup              = mysql.connection.cursor()
        selSup.execute("SELECT * FROM supervisor WHERE idSup=%s",(idSup,))
        sup                 = selSup.fetchone()
       

        nombr = sup.get('nombreSup')
        nombr = nombr.encode()
        nombr = encriptar.decrypt(nombr)
        nombr = nombr.decode()

        
        # SE MANDA EL CORREO
        msg = Message('Código de verificación', sender=PCapp.config['MAIL_USERNAME'], recipients=[correoSup])
        msg.body = render_template('layoutmail.html', name=nombr, verification_code=codVeriSup)
        msg.html = render_template('layoutmail.html', name=nombr, verification_code=codVeriSup)
        mail.send(msg)       

        flash('Cuenta creada con exito.')

        #MANDAR A UNA VENTANA PARA QUE META EL CODIGO DE VERFICIACION
        return redirect(url_for('verSupervisor'))
    else:
        flash('Error al crear la cuenta.')
        return redirect(url_for('verSupervisor'))

#~~~~~~~~~~~~~~~~~~~ Ver Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerSupervisor', methods=['GET', 'POST'])
def verSupervisor():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginAdmin' in session:
                # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
                encriptar = encriptado()

                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                list_consult = ['supervisor', 'activoSup']
                list_campo = ['nombreSup', 'apellidoPSup', 'apellidoMSup']
                
                # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
                sup, datosSup = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

                return render_template('adm_super.html', super = sup, datosSup = datosSup, username=session['name'], email=session['correoAd'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))

    
#~~~~~~~~~~~~~~~~~~~ Editar Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaSupervisor', methods=["GET", "POST"])
def editarCuentaSupervisor():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombreSup', 'apellidoPSup', 'apellidoMSup']
    list_campos_consulta = ['idSup', 'supervisor', 'nombreSup', 'apellidoPSup', 'apellidoMSup']
    
    # SE RECIBE LA INFORMACION
    nombreSupCC , apellidoPSupCC , apellidoMSupCC =  get_information_3_attributes(encriptar, request, list_campos)
   
    consult_edit(request, mysql, list_campos_consulta, nombreSupCC, apellidoPSupCC, apellidoMSupCC)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verSupervisor'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaSupervisor', methods=["GET", "POST"])
def eliminarCuentaSupervisor():
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['idSup', 'supervisor', 'activoSup']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verSupervisor'))


@PCapp.route('/IndexAdministrador', methods=['GET', 'POST'])
def indexAdministrador():
    if not 'login' in session:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))

    if not session['verificado'] == 2:
        flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
        return redirect(url_for('verify'))
    
    if not 'loginAdmin' in session:
        flash("No tienes permiso para acceder a esta página", 'danger')
        return redirect(url_for('home'))
    
    return render_template('index_admin.html', username=session['name'], email=session['correoAd'])
    

#~~~~~~~~~~~~~~~~~~~ Index Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/IndexPracticantes', methods=["GET", "POST"])
def indexPracticantes():
    if 'login' in session:
        if session['verificado'] == 2:
            if 'loginPrac' in session:
                #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
                encriptar = encriptado()

                idPrac = session['idPrac']
                
                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                list_campo  = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
                list_consult = [idPrac, 1]
                
                pra, datosPrac = obtener_datos(list_campo, list_consult, mysql, encriptar, 3)

                # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
                list_cosult = [idPrac, 4]
                praH, datosPracH = obtener_datos(list_campo, list_consult, mysql, encriptar, 3)

                return render_template('index_practicante.html', pract = pra, datosPrac = datosPrac, datosPracH=datosPracH, username=session['name'], email=session['correoPrac'])
            else:
                flash("No tienes permiso para acceder a esta página", 'danger')
                return redirect(url_for('home'))
        else:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
    else:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))
    


@PCapp.route('/')
@login_required
@verified_required
def home():
    if 'loginPaci' in session:
        return redirect(url_for('indexPacientes'))
    elif 'loginPrac' in session:
        return redirect(url_for('indexPracticantes'))
    elif 'loginSup' in session:
        return redirect(url_for('indexSupervisor'))
    elif 'loginAdmin' in session:
        return redirect(url_for('indexAdministrador'))
    else:
        return redirect(url_for('auth'))

@PCapp.route('/AgregarPracticante')
@login_required
@verified_required
def agregarPracticante():
    if not 'login' in session:
        flash("Por favor, inicia sesión para continuar", 'warning')
        return redirect(url_for('auth'))
         
    if not session['verificado'] == 2: 
        flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
        return redirect(url_for('verify')) 
       
    if not 'loginSup' in session:
        flash("No tienes permiso para acceder a esta página", 'danger')
        return redirect(url_for('home'))
    
    return render_template('agregar_practicante.html', username=session['name'], email=session['correoSup'])

@PCapp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth'))

@PCapp.route('/protected')
@verified_required
def protected ():
    return "<h1>Esta es una vista protegida, solo para usuarios autenticados.</h1>"

def status_401(error):
    return redirect(url_for('login'))

def status_404(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    PCapp.secret_key = '123'
    csrf.init_app(PCapp)
    PCapp.register_error_handler(401,status_401)
    PCapp.register_error_handler(404,status_404)
    PCapp.run(port=3000,debug=True)
    
    
#E we, cuanto es 2 + 2?  = △⃒⃘

# FALTA PROBAR CITAS EN PACIENTES
# ELIMINAR CITAS (PACIENTES Y PRACTICANTES)
# ALL EL SISTEMA DE ENCUESTAS ZZZZZZZ