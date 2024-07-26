from __future__ import print_function

from ast                    import If
from threading              import activeCount
from time                   import time
from flask                  import Flask, render_template, request, redirect, url_for, session, flash, make_response, jsonify
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
PCapp.config['MYSQL_PASSWORD']          = 'root'
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


##### FUNCIONES DECORADOR
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'login' not in session:
            flash("Por favor, inicia sesión para continuar", 'warning')
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

def already_logged_in(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'login' in session:
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def verified_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Asumiendo que el estado de verificación se guarda en la sesión del usuario
        if 'verificado' not in session or not session['verificado'] or session['verificado'] != 2:
            flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
            return redirect(url_for('verify'))
        return f(*args, **kwargs)
    return decorated_function


# if 'verificado' not in session or session['verificado'] != 2:
#     flash("Por favor, verifica tu cuenta antes de continuar", 'warning')
#     return redirect(url_for('verify'))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loginAdmin' not in session:
            flash("No tienes permiso para acceder a esta página", 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def supervisor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loginSup' not in session:
            flash("No tienes permiso para acceder a esta página", 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def practicante_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loginPrac' not in session:
            flash("No tienes permiso para acceder a esta página", 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def paciente_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loginPaci' not in session:
            flash("No tienes permiso para acceder a esta página", 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Diagnostico requerido
def survey_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'survey' not in session or session['survey'] != 1:
            return redirect(url_for('survey_v2'))
        return f(*args, **kwargs)
    return decorated_function

# Diagnostico hecho
def done_survey_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if  session['survey'] == 1:
            flash("Diagnostico ya hecho", 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def require_post(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method != 'POST':
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function
##################################################################3

#El Def auth controla si inicia sesion o es nuevo usuario

@PCapp.route('/pythonlogin/', methods=['GET', 'POST'])
@already_logged_in
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
                'supervisor'    : ['supervisor',   'correoSup',   'activoSup',   'veriSup',   1,  'contraSup',   'loginSup',   'idSup',   'nombreSup',   'indexSupervisor'],
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
            
            cur.close()
            selPaci.close()           
            
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
@login_required
@verified_required
@paciente_required
@done_survey_required
def survey_v2():
    return render_template('/paci/SurveyV2modcopy.html')


@PCapp.route('/results', methods=['POST'])
@require_post
def results():
        reported_symptoms = [symptom for symptom, value in request.form.items() if value == 'si']
        print(f"{reported_symptoms}")
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
    if 'login' not in session:
        return redirect(url_for('auth'))
    
    if session['verificado'] == 2:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        user_code = request.form['code']
        flash("Revisa tu correo electrónico para obtener tu código de verificación", 'success')

        # DICCIONARIO PARA EL MANEJO DE DATOS DE VERIFICACION
        dicc_verify = {
            'paciente':    ["paciente",    "codVeriPaci", "veriPaci", "activoPaci"],
            'practicante': ["practicante", "codVeriPrac", "veriPrac", "activoPrac"],
            'supervisor':  ["supervisor",  "codVeriSup",  "veriSup",  "activoSup"],
            'admin':       ["admin",       "codVeriAd",   "veriAd",   "activoAd"]
        }

        # Verificar el código de verificación
        if verify_code(mysql, session, dicc_verify, user_code):
            flash("Registro completado con éxito", 'success')
            return redirect(url_for('auth'))

        flash("Código de verificación incorrecto", 'danger')

    return render_template('verify.html')  
 
 
def encriptado():
    with  mysql.connection.cursor() as selectoken:
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
@login_required
@verified_required
@admin_required
def crearCuentaAdmin():
   
    if request.method == 'POST':
        campos_validacion_ad = {
            'nombreAd':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
            'apellidoPAd': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'apellidoMAd': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'correoAd': {'max_length': 35, 'pattern': r"^[a-z]+\.[a-z]+\d{4}@(alumnos\.udg\.mx|udg\.com\.mx|academicos\.udg\.mx)$"},
            'contraAd': {'pattern': r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+}{'?/:;.,]).{8,}"}
        }

        etiquetas_ad = {
            'nombreAd': 'Nombre',
            'apellidoPAd': 'Apellido Paterno',
            'apellidoMAd': 'Apellido Materno',
            'correoAd': 'Correo',
            'contraAd': 'Contraseña'
        }
        errores = validar_campos(request, campos_validacion_ad, etiquetas_ad)

        if errores:
            for error in errores:
                flash(error)
            return redirect(url_for('verAdministrador'))
       
       
        # CONFIRMAR CORREO CON LA BD
        correoAd        = request.form['correoAd']

        contraAd        = request.form['contraAd']
        
        #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
        encriptar = encriptado()
        
        # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
        list_campos = ['nombreAd', 'apellidoPAd', 'apellidoMAd']
        
        # SE RECIBE LA INFORMACION
        nombreAdCC, apellidoPAdCC , apellidoMAdCC = get_information_3_attributes(encriptar, request, list_campos)
        
        hashed_password = bcryptObj.generate_password_hash(contraAd).decode('utf-8')

        # CODIGO DE SEGURIDAD 
        codVeriAd   = security_code()
        activoAd    = 0
        veriAd      = 1
        priviAd     = 1
        
        
        # Verificar si el correo ya está registrado en la base de datos
        with mysql.connection.cursor() as cur:
            cur.execute("SELECT * FROM admin WHERE correoAd=%s AND activoAd IS NOT NULL", [correoAd])
            if cur.rowcount > 0:
                flash("El correo ya está registrado", 'danger')
                return redirect(url_for('verAdministrador'))

        try:
            with mysql.connection.cursor() as regAdmin:
                regAdmin.execute(
                    """
                    INSERT INTO admin 
                    (nombreAd, apellidoPAd, apellidoMAd, correoAd, contraAd, codVeriAd, activoAd, veriAd, priviAd) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (nombreAdCC, apellidoPAdCC, apellidoMAdCC, correoAd, hashed_password, codVeriAd, activoAd, veriAd, priviAd)
                )
                mysql.connection.commit()
                idAd = regAdmin.lastrowid
        
            with mysql.connection.cursor() as selAd:
                selAd.execute("SELECT nombreAd FROM admin WHERE idAd=%s", (idAd,))
                ad = selAd.fetchone()
                
                
            nombre_administrador = encriptar.decrypt(ad['nombreAd'].encode()).decode()

        
            # SE MANDA EL CORREO
            msg = Message(
                'Código de verificación', 
                sender=PCapp.config['MAIL_USERNAME'], 
                recipients=[correoAd]
            )
            msg.body = render_template('layoutmail.html', name=nombre_administrador, verification_code=codVeriAd)
            msg.html = render_template('layoutmail.html', name=nombre_administrador, verification_code=codVeriAd)
            mail.send(msg)
            

            flash("Revisa tu correo electrónico para ver los pasos para completar tu registro!", 'success')        
            #MANDAR A UNA VENTANA PARA QUE META EL CODIGO DE VERFICIACION
            return redirect(url_for('verAdministrador'))
        
        except Exception as e:
            mysql.connection.rollback()
            flash("Ocurrió un error al crear la cuenta: " + str(e), 'danger')
            return redirect(url_for('verAdministrador'))
        
    else:
        flash("Error al crear la cuenta", 'danger')
        return redirect(url_for('verAdministrador'))  

    
#~~~~~~~~~~~~~~~~~~~ Ver Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerAdministrador', methods=['GET', 'POST'])
@login_required
@verified_required
@admin_required
def verAdministrador():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['admin', 'activoAd']
    list_campo = ['nombreAd', 'apellidoPAd', 'apellidoMAd']
    
    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
    ad, datosAd = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

    return render_template('/adm/adm_adm.html', admin = ad, datosAd = datosAd, username=session['name'], email=session['correoAd'])    

#~~~~~~~~~~~~~~~~~~~ Eliminar Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaAdmin', methods=["GET", "POST"])
@login_required
@verified_required
@admin_required
@require_post
def eliminarCuentaAdmin():
    list_consult = ['idAd', 'admin', 'activoAd']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta eliminada con exito.')
    return redirect(url_for('verAdministrador'))


#~~~~~~~~~~~~~~~~~~~ Editar Adminsitradores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaAdmin', methods=["GET", "POST"])
@login_required
@verified_required
@admin_required
@require_post
def editarCuentaAdmin():
    
    # Validacion de campos
    campos_validacion = {
        'nombreAd':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
        'apellidoPAd': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
        'apellidoMAd': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"}
    }
    etiquetas = {
        'nombreAd': 'Nombre',
        'apellidoPAd': 'Apellido Paterno',
        'apellidoMAd': 'Apellido Materno'
     }
    
    errores = validar_campos(request, campos_validacion, etiquetas)
    
    if errores:
        for error in errores:
            flash(error)
        return redirect(url_for('verAdministrador'))
    
    
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
@PCapp.route('/IndexPacientes', methods=['GET', 'POST'])
@login_required
@verified_required
@paciente_required
@survey_required
def indexPacientes():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
    idPaci = session['idPaci']

    # FALTA PROBAR ESTO
    with mysql.connection.cursor() as selecCita:
        selecCita.execute("""
            SELECT * FROM citas C 
            INNER JOIN practicante PR ON C.idCitaPrac = PR.idPrac 
            INNER JOIN paciente PA ON C.idCitaPaci = PA.idPaci 
            WHERE idCitaPaci=%s AND estatusCita=%s
        """,(idPaci,1))
        cit = selecCita.fetchone()

    if cit:
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

    return render_template('/paci/index_pacientes.html', hc = hc, datosCitas = datosCitas, cit = cit, citaRealizada = citaRealizada, username=session['name'], email=session['correoPaci'], request=request)


######################## Falta
########################
########################
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

##########################################
##########################################
##########################################

@PCapp.route('/VerPacientesAdm', methods=['GET', 'POST'])
@login_required
@verified_required
@admin_required
def verPacientesAdm():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['paciente', 'activoPaci']
    list_campo = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']

    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
    pac, datosPaci = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

    return render_template('/adm/adm_pacie.html', paci = pac, datosPaci = datosPaci, username=session['name'], email=session['correoAd'])


@PCapp.route('/CrearCita', methods=["POST"])
@csrf.exempt
@login_required
@verified_required
@paciente_required
def crearCita():
    if request.headers.get('X-Requested-With') != 'XMLHttpRequest':
        return jsonify({'success': False, 'message': 'Solicitud no válida'}), 400

    try:
        # Obtener datos del paciente desde la sesión
        id_paci = session['idPaci']
        correo_paci = session['correoPaci']
        
        # Recuperar datos del formulario
        id_prac = request.form['idPrac']
        correo_prac = request.form['correoPrac']
        tipo_cita = request.form['tipoCita']
        fecha_hora_cita = request.form['fechaHoraCita']
        
        # Formatear fecha y hora
        fecha_hora = datetime.datetime.strptime(fecha_hora_cita, '%Y-%m-%dT%H:%M')
        fecha = fecha_hora.date()
        hora = fecha_hora.time()

        dire_cita = "Modulo X" if tipo_cita == "Presencial" else "Virtual"
        
        estatus_cita = 1
        
        # Configuración de Google Calendar
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                
        service = build('calendar', 'v3', credentials=creds)
        event = crear_evento(dire_cita, tipo_cita, fecha_hora, correo_prac, correo_paci)

        if dire_cita == "Virtual":
            event['conferenceData'] = {
                'createRequest': {
                    'requestId': security_code(),
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            }
            event = service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1,
                sendNotifications=True
            ).execute()
        else:
            event = service.events().insert(
                calendarId='primary',
                body=event,
                sendNotifications=True
            ).execute()
        
        evento_id = event["id"]
        print("Esta es la fecha:",fecha)
        print("Esta es la hora:",hora)

        # Actualizar estado de la cita en la base de datos
        with mysql.connection.cursor() as cursor:
            cursor.execute("UPDATE paciente SET citaActPaci=%s WHERE idPaci=%s", (estatus_cita, id_paci))
            cursor.execute("UPDATE practicante SET estatusCitaPrac=%s WHERE idPrac=%s", (estatus_cita, id_prac))
            cursor.execute("""
                INSERT INTO citas (tipo, correoCitaPra, correoCitaPac, direCita, fechaCita, horaCita, estatusCita, eventoId, idCitaPrac, idCitaPaci) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (tipo_cita, correo_prac, correo_paci, dire_cita, fecha, hora, estatus_cita, evento_id, id_prac, id_paci))
            hora = hora.strftime('%H:%M')
            cursor.execute("""
                UPDATE horario SET permitido=%s WHERE fecha=%s AND hora=%s AND practicante_id=%s
            """, (0, fecha, hora, id_prac))
            mysql.connection.commit()

        return jsonify({'success': True, 'message': 'Cita agendada con éxito.'})

    except Exception as e:
        print(f'An error occurred: {str(e)}')
        return jsonify({'success': False, 'message': f'Ocurrió un error al crear la cita: {str(e)}'}), 400

        
#~~~~~~~~~~~~~~~~~~~ Contestar Encuesta ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EncuestaPaciente', methods=["GET"])
@login_required
@verified_required
@paciente_required
def encuestaPaciente():
    # Crear instancia de encriptación
    encriptar = encriptado()

    # Obtener el ID del paciente de la sesión
    id_paci = session['idPaci']
    
    datos_citas = []

    # Ejecutar la consulta para obtener la cita actual del paciente
    with mysql.connection.cursor() as cursor:
        query = """
            SELECT * 
            FROM citas C 
            INNER JOIN practicante PR ON C.idCitaPrac = PR.idPrac 
            INNER JOIN paciente PA ON C.idCitaPaci = PA.idPaci 
            WHERE C.idCitaPaci = %s AND C.estatusCita = %s
        """
        cursor.execute(query, (id_paci, 1))
        cita = cursor.fetchone()

    if cita:
        # Campos de la cita que necesitamos
        list_campos = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
        
        # Decodificar los campos de la cita
        cita_decoded = select_and_decode_atribute(cita, list_campos, encriptar)
        
        # Actualizar la cita con los campos decodificados
        cita.update(cita_decoded)
        
        # Añadir la cita a la lista de datos
        datos_citas.append(cita)

        # Desempaquetar datos para el renderizado
        id_cita = cita.get('idCita')
        id_prac = cita.get('idPrac')
        nombre_prac = cita_decoded.get('nombrePrac')
        apellido_p_prac = cita_decoded.get('apellidoPPrac')
        apellido_m_prac = cita_decoded.get('apellidoMPrac')
    else:
        # Manejo de caso en que no se encuentra la cita
        id_cita = id_prac = nombre_prac = apellido_p_prac = apellido_m_prac = None

    # Renderizar la plantilla con los datos obtenidos
    return render_template('/paci/encuesta_paciente.html', nombrePrac=nombre_prac, apellidoPPrac=apellido_p_prac, apellidoMPrac=apellido_m_prac, idCita=id_cita, idPrac=id_prac, username=session['name'], email=session['correoPaci'])


# ~~~~~~~~~~~~~~~~~~~ Respuestas Encuesta ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/ContestarEncuesta', methods=["GET", "POST"])
@login_required
@verified_required
@paciente_required
def contestarEncuesta():
    # Obtener el ID del paciente desde la sesión
    id_paci = session['idPaci']
    
    # Obtener los datos del formulario
    id_prac = request.form.get('idPrac')
    id_cita = request.form.get('idCita')
    respuestas = [request.form.get(f'calificacion-{i}') for i in range(1, 9)]

    # Validar que todas las respuestas están presentes
    if not all(respuestas):
        flash('Por favor, conteste todas las preguntas de la encuesta.', 'danger')
        return redirect(url_for('indexPacientes'))

    # Obtener el ID del supervisor asociado al practicante
    try:
        with mysql.connection.cursor() as cursor:
            cursor.execute("SELECT idSupPrac FROM practicante WHERE idPrac = %s", (id_prac,))
            supervisor_id = cursor.fetchone()['idSupPrac']
    except mysql.connector.Error as err:
        flash(f'Error al obtener el ID del supervisor: {err}', 'danger')
        return redirect(url_for('indexPacientes'))

    # Insertar los datos de la encuesta en la base de datos
    try:
        with mysql.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO encuesta (
                    pregunta1Encu, pregunta2Encu, pregunta3Encu, pregunta4Encu,
                    pregunta5Encu, pregunta6Encu, pregunta7Encu, pregunta8Encu,
                    idEncuPaci, idEncuPrac, idSup
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (*respuestas, id_paci, id_prac, supervisor_id))
            mysql.connection.commit()

            # Obtener el ID de la encuesta insertada
            id_encuesta = cursor.lastrowid

            # Actualizar el estatus de la cita con el ID de la encuesta
            cursor.execute("""
                UPDATE citas 
                SET idEncuestaCita = %s, estatusCita = %s 
                WHERE idCita = %s
            """, (id_encuesta, 4, id_cita))
            mysql.connection.commit()
        
    except mysql.connector.Error as err:
        flash(f'Error al procesar la encuesta: {err}', 'danger')
        return redirect(url_for('indexPacientes'))

    # Confirmar éxito y redirigir
    flash('Encuesta contestada con éxito.', 'success')
    return redirect(url_for('indexPacientes'))


# ~~~~~~~~~~~~~~~~~~~ Calendario V1 ~~~~~~~~~~~~~~~~~~~#
# @PCapp.route('/Calendario/<string:idPrac>', methods=['GET'])
# def calendario(idPrac):
#     with mysql.connection.cursor() as cursor:
#         cursor.execute("DELETE FROM horario WHERE fecha < CURDATE()")
#         mysql.connection.commit()
#         cursor.execute('SELECT idPrac, correoPrac FROM practicante WHERE idPrac = %s', (idPrac,))
#         practicante = cursor.fetchone()
    
#     if practicante:
#         return render_template('calendario.html', idPrac=practicante['idPrac'],  correoPrac=practicante['correoPrac'], username=session['name'], email=session['correoPaci'])


def generar_horarios(practicante_id, mysql, num_days=21):
    # Fecha de inicio
    start_date = datetime.datetime.now()

    # Generar fechas y horarios para insertar
    try:
        with mysql.connection.cursor() as cursor:
            for day in range(num_days):
                current_date = start_date + datetime.timedelta(days=day)
                
                # Verificar si el día actual es un día de la semana (lunes a viernes)
                if current_date.weekday() < 5:  # 0-4 representa lunes a viernes
                    for hour in range(8, 21):  # Horas de 08:00 a 20:00
                        # Formatear fecha y hora
                        fecha = current_date.date()
                        hora = f"{hour:02d}:00"
                        
                        # Si es el día actual, evitar horas que ya pasaron
                        if current_date.date() == datetime.datetime.now().date() and hour <= datetime.datetime.now().hour:
                            continue
                        
                        # Verificar si ya existe un registro con la misma fecha, hora y practicante
                        cursor.execute("""
                            SELECT COUNT(*) as count FROM horario
                            WHERE fecha = %s AND hora = %s AND practicante_id = %s
                        """, (fecha, hora, practicante_id))
                        result = cursor.fetchone()
                        
                        # Insertar en la tabla horario si no existe ya un registro
                        if result['count'] == 0:
                            cursor.execute("""
                                INSERT INTO horario (fecha, hora, permitido, practicante_id)
                                VALUES (%s, %s, %s, %s)
                            """, (fecha, hora, 1, practicante_id))  # Suponiendo que 'permitido' es 1 (disponible)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Confirmar los cambios
        mysql.connection.commit()

# ~~~~~~~~~~~~~~~~~~~ Calendario V2 CON HORAS ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/Calendario/<string:idPrac>', methods=['GET'])
def calendario(idPrac):
    with mysql.connection.cursor() as cursor:
        # Eliminar registros anteriores a la fecha y hora actual
        cursor.execute("DELETE FROM horario WHERE fecha < CURDATE() OR (fecha = CURDATE() AND hora < CURTIME())")
        mysql.connection.commit()
        
        # Verificar si ya existen horarios para este practicante en el futuro
        cursor.execute("""
            SELECT COUNT(*) as count FROM horario 
            WHERE practicante_id = %s AND (fecha > CURDATE() OR (fecha = CURDATE() AND hora > CURTIME()))
        """, (idPrac,))
        result = cursor.fetchone()
        
        # Solo generar horarios si no existen registros futuros para este practicante
        if result['count'] == 0:
            generar_horarios(idPrac, mysql)

        cursor.execute('SELECT idPrac, correoPrac FROM practicante WHERE idPrac = %s', (idPrac,))
        practicante = cursor.fetchone()
    
    if practicante:
        return render_template('calendario.html', idPrac=practicante['idPrac'],  correoPrac=practicante['correoPrac'], username=session['name'], email=session['correoPaci'])

@PCapp.route('/Horario/<string:idPrac>', methods=['GET'])
def obtener_horarios(idPrac):
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT fecha, hora, permitido FROM horario WHERE practicante_id = %s", (idPrac,))
        horarios = cursor.fetchall()
        horarios_dict = [{'fecha': row['fecha'].strftime('%Y-%m-%d'), 'hora': row['hora'], 'permitido': row['permitido']} for row in horarios]
        return jsonify(horarios_dict)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    finally:
        cursor.close()


# ~~~~~~~~~~~~~~~~~~~ Ver Encuestas de Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerEncuestasPracticante/<string:idPrac>', methods=['GET', 'POST'])
@login_required
@verified_required
@supervisor_required
def verEncuestasPracticante(idPrac):
    # USAR SESSION PARA OBTENER EL ID DE SUPERVISOR
    id_Sup = session['idSup']
    
    # Verificar que el practicante pertenece al supervisor
    try:
        with mysql.connection.cursor() as cursor:
            cursor.execute("""
                SELECT *
                FROM practicante
                WHERE idPrac = %s AND idSupPrac = %s
            """, (idPrac, id_Sup))
            if cursor.fetchone() is None:
                flash("No tienes permiso para ver estas encuestas.")
                return redirect(url_for('home'))
    except Exception as e:
        print(f"Error al verificar el practicante: {e}")
        flash("Hubo un problema al verificar el practicante.")
        return redirect(url_for('home'))

    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # Definición de listas para consulta de datos
    list_consult = ['encuesta', 'E', 'E.idEncuPrac', 'E.idEncuPaci', 'idEncuPrac', idPrac]
    list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']


    encu, datosEncu = obtener_datos(list_campo, list_consult, mysql, encriptar, 2)
    
    return render_template('encuesta_practicante.html', datosEncu = datosEncu, username=session['name'], email=session['correoSup'])


# ~~~~~~~~~~~~~~~~~~~ Ver Resultados de Encuestas ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerResultadosEncuesta/<string:idEncu>', methods=['GET'])
@login_required
@verified_required
@supervisor_required
def verResultadosEncuesta(idEncu):
    # Obtener el ID del supervisor desde la sesión
    id_sup = session['idSup']
    
    try:
        # Seleccionar el resultado de la encuesta y verificar que pertenezca al supervisor
        with mysql.connection.cursor() as cursor:
            cursor.execute("""
                SELECT * 
                FROM encuesta 
                WHERE idEncu = %s AND idSup = %s
            """, (idEncu, id_sup))
            encuesta = cursor.fetchone()
        
        if not encuesta:
            flash("No tienes permiso para ver estos resultados.", 'danger')
            return redirect(url_for('home'))

        return render_template('resultados_encuestas.html', resu=encuesta, username=session['name'], email=session['correoSup'])
    
    except mysql.connector.Error as err:
        flash(f'Error al obtener los resultados de la encuesta: {err}', 'danger')
        return redirect(url_for('home'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Paciente ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaPaciente', methods=["POST"])
@require_post
def eliminarCitaPaciente():
    try:
        # Recuperar datos del formulario
        idCita = request.form['idCita']
        fechaCita = request.form['fechaCita']
        horaCita = request.form['horaCita']
        eventoIdCita = request.form['eventoCita']
        
        fecha_hora_cita = datetime.datetime.strptime(f"{fechaCita} {horaCita}", '%Y-%m-%d %H:%M:%S')
        
        # Obtener la fecha y hora actual
        fecha_hora_actual = datetime.datetime.now()
        
        # Calcular la diferencia entre la fecha y hora de la cita y la fecha y hora actual
        diferencia = fecha_hora_cita - fecha_hora_actual

        # Verificar si hay más de 24 horas antes de la cita
        if diferencia.total_seconds() > 24 * 3600:
            # Configurar credenciales de Google Calendar API
            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            # Eliminar evento de Google Calendar
            service = build('calendar', 'v3', credentials=creds)
            service.events().delete(calendarId='primary', eventId=eventoIdCita, sendNotifications=True).execute()
            
            # Actualizar estatus de la cita en la base de datos
            with mysql.connection.cursor() as cursor:
                cursor.execute("SELECT idCitaPrac FROM citas WHERE idCita = %s", (idCita,))
                result = cursor.fetchone()
                if result:
                    id_prac = result['idCitaPrac']
                else:
                    flash('No se encontró la cita.')
                    return redirect(url_for('indexPacientes'))

                # Actualizar el estatus de la cita en la base de datos
                cursor.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s", (2, idCita))
                
                # Actualizar la tabla horario
                horaCita_obj = datetime.datetime.strptime(str(horaCita), '%H:%M:%S').time()
                horaCita_formateada = horaCita_obj.strftime('%H:%M')
                cursor.execute("UPDATE horario SET permitido=%s WHERE fecha=%s AND hora=%s AND practicante_id=%s", 
                   (1, fechaCita,  horaCita_formateada, id_prac))

                mysql.connection.commit()
            
            flash('Cita eliminada con éxito.')
        else:
            flash("No se puede cancelar la cita, faltan menos de 24 horas")
    
    except Exception as e:
        print(f"Error: {e}")
        flash("Hubo un problema al eliminar la cita.")

    return redirect(url_for('indexPacientes'))


#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Supervisor ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaSupervisor', methods=["POST"])
@require_post
def eliminarCitaSupervisor():
    try:
        # Recuperar datos del formulario
        idCita = request.form['idCita']
        eventoIdCita = request.form['eventoCita']
        cancelacion = request.form['cancelacion']

        if cancelacion == 'Si':
            # Configurar credenciales de Google Calendar API
            print("Como no pa, ahi te va tu cancelacion.")

            creds = None
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                    creds = flow.run_local_server(port=0)
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())

            # Eliminar evento de Google Calendar
            service = build('calendar', 'v3', credentials=creds)
            service.events().delete(calendarId='primary', eventId=eventoIdCita, sendNotifications=True).execute()

            # Actualizar estatus de la cita en la base de datos 
            with mysql.connection.cursor() as cursor:
                cursor.execute("SELECT idCitaPrac, fechaCita, horaCita FROM citas WHERE idCita = %s", (idCita,))
                result = cursor.fetchone()
                if result:
                    id_prac   = result['idCitaPrac']
                    fechaCita = result['fechaCita']
                    horaCita  = result['horaCita']
                else:
                    flash('No se encontró la cita.')
                    return redirect(url_for('home'))

                # Actualizar el estatus de la cita en la base de datos
                cursor.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s", (2, idCita))
                
                # Actualizar la tabla horario
                horaCita_obj = datetime.datetime.strptime(str(horaCita), '%H:%M:%S').time()
                horaCita_formateada = horaCita_obj.strftime('%H:%M')
                cursor.execute("UPDATE horario SET permitido=%s WHERE fecha=%s AND hora=%s AND practicante_id=%s", 
                   (1, fechaCita,  horaCita_formateada, id_prac))

                mysql.connection.commit()
            
            flash('Cita eliminada con éxito.')
        else:
            cursor = mysql.connection.cursor()
            cursor.execute("UPDATE citas SET estatusCita=%s WHERE idCita=%s", (1, idCita))
            mysql.connection.commit()
            cursor.close()
            
            flash("No se puede cancelar la cita, faltan menos de 24 horas")
    
    except Exception as e:
        print(f"Error: {e}")
        flash("Hubo un problema al eliminar la cita.")

    return redirect(url_for('indexSupervisor'))
    

# ~~~~~~~~~~~~~~~~~~~ Crear Cita ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/AgendarCita', methods=['GET', 'POST'])
@login_required
@verified_required
@paciente_required
def agendarCita():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # USAR EL SESSION PARA OBTENER EL ID DEL PACIENTE
    idPaci = session['idPaci']

    # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
    with mysql.connection.cursor() as selecPrac:
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
        
    return render_template('/paci/agenda_cita copy.html', datosPrac = datosPrac, username=session['name'], email=session['correoPaci'])



#~~~~~~~~~~~~~~~~~~~ Eliminar Cita Pracicante ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCitaPracticante', methods=["GET", "POST"])
@require_post
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
        with mysql.connection.cursor() as editarCita:
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
@PCapp.route('/IndexSupervisor', methods=['GET', 'POST'])
@login_required
@verified_required
@supervisor_required
def indexSupervisor():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()
    
    # USAR SESSION PARA OBTENER EL ID DE SUPERVISOR
    idSup = session['idSup']

    # SE SELECCIONA TODOS LOS DATOS DE LA BD POR SI SE LLEGA A NECESITAR
    with mysql.connection.cursor() as selecPrac:
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
    with mysql.connection.cursor() as selecCitas:
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


    return render_template('/sup/index_supervisor.html', pract = pra, datosPrac = datosPrac, datosCitas = datosCitas, username=session['name'], email=session['correoSup'], request=request)



#~~~~~~~~~~~~~~~~~~~ Ver Practicantes ADMINISTRADOR ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/VerPracticantesAdm', methods=['GET', 'POST'])
@login_required
@verified_required
@admin_required
def verPracticantesAdm():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['practicante', ' activoPrac']
    list_campo = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']

    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
    pra, datosPrac = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

    return render_template('/adm/adm_pract.html', pract = pra, datosPrac = datosPrac, username=session['name'], email=session['correoAd'])

    

#~~~~~~~~~~~~~~~~~~~ Eliminar Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPracticantesAdm', methods=["GET", "POST"])
@login_required
@verified_required
@admin_required
@require_post
def eliminarCuentaPracticantesAdm():

    list_consult = ['idPrac', 'practicante', 'activoPrac']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesAdm'))

    #~~~~~~~~~~~~~~~~~~~ Eliminar Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPracticantesSup', methods=["GET", "POST"])
@require_post
def eliminarCuentaPracticantesSup():

    list_consult = ['idPrac', 'practicante', 'activoPrac']
    eliminarCuenta(request, mysql, list_consult)
    
    flash('Cuenta editada con exito.')
    return redirect(url_for('indexSupervisor'))

#~~~~~~~~~~~~~~~~~~~ Editar Practicantes ADMIN ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPracticantesAdm', methods=["GET", "POST"])
@require_post
def editarCuentaPracticantesAdm():

    # Validacion de campos
    campos_validacion = {
        'nombrePrac':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
        'apellidoPPrac': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
        'apellidoMPrac': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"}
    }
    etiquetas = {
        'nombrePrac': 'Nombre',
        'apellidoPPrac': 'Apellido Paterno',
        'apellidoMPrac': 'Apellido Materno'
     }
    
    errores = validar_campos(request, campos_validacion, etiquetas)
    
    if errores:
        for error in errores:
            flash(error)
        return redirect(url_for('verPracticantesAdm'))
    
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()
    
    idPrac               = request.form['idPrac']

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campos = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    list_campos_consulta = ['idPrac', 'practicante', 'nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
    
    # SE RECIBE LA INFORMACION
    nombrePracCC, apellidoPPracCC, apellidoMPracCC = get_information_3_attributes(encriptar, request, list_campos)
    
    consult_edit(request,mysql,list_campos_consulta, nombrePracCC, apellidoPPracCC, apellidoMPracCC)
    
    flash('Cuenta editada con exito.')
    return redirect(url_for('verPracticantesAdm'))

#~~~~~~~~~~~~~~~~~~~ Editar Practicantes Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPracticantesSup', methods=["GET", "POST"])
@require_post
def editarCuentaPracticantesSup():

    # Validacion de campos
    campos_validacion = {
        'nombrePrac':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
        'apellidoPPrac': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
        'apellidoMPrac': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"}
    }
    etiquetas = {
        'nombrePrac': 'Nombre',
        'apellidoPPrac': 'Apellido Paterno',
        'apellidoMPrac': 'Apellido Materno'
     }
    
    errores = validar_campos(request, campos_validacion, etiquetas)
    
    if errores:
        for error in errores:
            flash(error)
        return redirect(url_for('indexSupervisor'))
    
    
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
        with mysql.connection.cursor() as picture:
            picture.execute("UPDATE practicante SET fotoPrac=%s WHERE idPrac=%s", (fotoActual, idPrac,))
            mysql.connection.commit()

    flash('Cuenta editada con exito.')
    return redirect(url_for('indexSupervisor'))


#~~~~~~~~~~~~~~~~~~~ Crear Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/CrearCuentaPracticantes', methods=["GET", "POST"])
@require_post
def crearCuentaPracticantes():
    if request.method == 'POST':
       
        campos_validacion_prac = {
            'nombrePrac':     {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
            'apellidoPPrac':  {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'apellidoMPrac':  {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'sexoPrac':       {'valid_options': ['Hombre', 'Mujer', 'NoDecirlo']},
            'fechaNacPrac':   {'pattern': r"\d{4}-\d{2}-\d{2}"},
            'celPrac':        {'max_length': 10, 'pattern': r"\d{10}"},
            'codigoUPrac':    {'max_length': 9, 'pattern': r"\d{9}"},
            'correoPrac':     {'max_length': 35, 'pattern': r"^[a-z]+\.[a-z]+\d{4}(@alumnos\.udg\.mx|@udg\.com\.mx|@academicos\.udg\.mx)$"},
            'contraPrac':     {'pattern': r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+{}\[\]:;,.]).{8,}"}
        }

        etiquetas_prac = {
            'nombrePrac': 'Nombre',
            'apellidoPPrac': 'Apellido Paterno',
            'apellidoMPrac': 'Apellido Materno',
            'sexoPrac': 'Sexo',
            'fechaNacPrac': 'Fecha de Nacimiento',
            'celPrac': 'Teléfono Celular',
            'codigoUPrac': 'Código',
            'correoPrac': 'Correo',
            'contraPrac': 'Contraseña'
        }
        errores = validar_campos(request, campos_validacion_prac, etiquetas_prac)
    
        if errores:
            for error in errores:
                flash(error)
            return redirect(url_for('agregarPracticante'))
        
        
        sexoPrac          = request.form['sexoPrac']
        fechaNacPrac      = request.form['fechaNacPrac']
        celPrac           = request.form['celPrac']
        codigoUPrac       = request.form['codigoUPrac']
        idSupPrac         = session['idSup']
        
        # CONFIRMAR CORREO CON LA BD
        correoPrac        = request.form['correoPrac']

        # CAMBIAR EL HASH DE LA CONTRA POR BCRYPT
        contraPrac        = request.form['contraPrac']
        
        
        #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
        encriptar = encriptado()

        # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
        list_campos = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac']
        
        # SE RECIBE LA INFORMACION
        nombrePracCC, apellidoPPracCC , apellidoMPracCC = get_information_3_attributes(encriptar, request, list_campos)
        
        hashed_password = bcryptObj.generate_password_hash(contraPrac).decode('utf-8')
        
        edad = date_to_age(fechaNacPrac)
       
        # CODIGO DE SEGURIDAD 
        codVeriPrac  = security_code()
        activoPrac   = 0
        veriPrac     = 1

        
         # Verificar si el correo ya está registrado en la base de datos
        with mysql.connection.cursor() as cur:
            result = cur.execute("SELECT * FROM practicante WHERE correoPrac=%s AND activoPrac IS NOT NULL", [correoPrac,])
            if result > 0:
                # Si el correo ya está registrado, mostrar un mensaje de error
                flash("El correo ya está registrado", 'danger')
                return redirect(url_for('indexSupervisor'))  
        
        with mysql.connection.cursor() as regPracticante:
            regPracticante.execute("INSERT INTO practicante (nombrePrac, apellidoPPrac, apellidoMPrac, contraPrac, sexoPrac, codVeriPrac, correoPrac, fechaNacPrac, activoPrac, veriPrac, edadPrac, celPrac, codigoUPrac, idSupPrac) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (nombrePracCC, apellidoPPracCC, apellidoMPracCC, hashed_password, sexoPrac, codVeriPrac, correoPrac, fechaNacPrac, activoPrac, veriPrac, edad, celPrac, codigoUPrac, idSupPrac,))
            mysql.connection.commit()

            idPrac              =   regPracticante.lastrowid
            
            # PARA SUBIR LA FOTO
        if request.files.get('foto'):
            foto                =   request.files['foto']
            fotoActual          =   secure_filename(foto.filename)
            foto.save(os.path.join(PCapp.config['UPLOAD_FOLDER'], fotoActual))
            with mysql.connection.cursor() as picture:
                picture.execute("UPDATE practicante SET fotoPrac=%s WHERE idPrac=%s", (fotoActual, idPrac,))
                mysql.connection.commit()

        # MANDAR CORREO CON CODIGO DE VERIRIFICACION
        with mysql.connection.cursor() as selPrac:
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
        return redirect(url_for('indexSupervisor'))
    else:
        flash('No se pudo crear la cuenta.')
        return redirect(url_for('indexSupervisor'))


# ~~~~~~~~~~~~~~~~~~~ Editar Pacientes Admin ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaPacienteAdm', methods=["GET", "POST"])
@require_post
def editarCuentaPacienteAdm():
    
    # Validacion de campos
    campos_validacion = {
        'nombrePaci':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
        'apellidoPPaci': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
        'apellidoMPaci': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"}
    }

    etiquetas = {
        'nombrePaci': 'Nombre',
        'apellidoPPaci': 'Apellido Paterno',
        'apellidoMPaci': 'Apellido Materno'
    }
    
    errores = validar_campos(request, campos_validacion, etiquetas)

    if errores:
        for error in errores:
            flash(error)
        return redirect(url_for('verPacientesAdm'))
    
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

#Esto para que sirve?
# ~~~~~~~~~~~~~~~~~~~ Editar Pacientes Supervisor ~~~~~~~~~~~~~~~~~~~#
# @PCapp.route('/EditarCuentaPacienteSup', methods=["GET", "POST"])
# @require_post
# def editarCuentaPacienteSup():
#     #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
#     encriptar = encriptado()
    
#     # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
#     list_campos = ['nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
#     list_campos_consulta = ['idPaci', 'paciente', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    
#     # SE RECIBE LA INFORMACION
#     nombrePaciCC, apellidoPPaciCC , apellidoMAdCC =  get_information_3_attributes(encriptar, request, list_campos)
   
#     consult_edit(request, mysql, list_campos_consulta, nombrePaciCC, apellidoPPaciCC, apellidoMAdCC)
    
#     flash('Cuenta editada con exito.')
#     return redirect(url_for('verPaciente'))

#~~~~~~~~~~~~~~~~~~~ Eliminar Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPacienteAdm', methods=["GET", "POST"])
@require_post
def eliminarCuentaPacienteAdm():
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['idPaci', 'paciente', 'activoPaci']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verPacientesAdm'))

#~~~~~~~~~~~~~~~~~~~ Eliminar Pacientes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EliminarCuentaPacienteSup', methods=["GET", "POST"])
@require_post
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
@require_post
def crearCuentaSupervisor():
    if request.method == 'POST':
        campos_validacion_sup = {
            'nombreSup':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
            'apellidoPSup': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'apellidoMSup': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
            'correoSup': {'max_length': 35, 'pattern': r"^[a-z]+\.[a-z]+\d{4}@(alumnos\.udg\.mx|udg\.com\.mx|academicos\.udg\.mx)$"},
            'contraSup': {'pattern': r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+}{'?/:;.,]).{8,}"}
        }

        etiquetas_sup = {
            'nombreSup': 'Nombre',
            'apellidoPSup': 'Apellido Paterno',
            'apellidoMSup': 'Apellido Materno',
            'correoSup': 'Correo',
            'contraSup': 'Contraseña'
        }
        errores = validar_campos(request, campos_validacion_sup, etiquetas_sup)

        if errores:
            for error in errores:
                flash(error)
            return redirect(url_for('verSupervisor'))

        # CONFIRMAR CORREO CON LA BD
        correoSup        = request.form['correoSup']

        contraSup        = request.form['contraSup']
        #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
        encriptar = encriptado()
        
        # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
        list_campos = ['nombreSup', 'apellidoPSup', 'apellidoMSup']
        
        # SE RECIBE LA INFORMACION
        nombreSupCC, apellidoPSupCC, apellidoMSupCC = get_information_3_attributes(encriptar, request, list_campos)
        
        hashed_password = bcryptObj.generate_password_hash(contraSup).decode('utf-8')

        # CODIGO DE SEGURIDAD
        codVeriSup  = security_code()
        activoSup   = 0
        veriSup     = 1
        priviSup    = 2
        
        # Verificar si el correo ya está registrado en la base de datos
        with mysql.connection.cursor() as cur:
            result = cur.execute("SELECT * FROM supervisor WHERE correoSup=%s AND activoSup IS NOT NULL", [correoSup,])
            if result > 0:
                # Si el correo ya está registrado, mostrar un mensaje de error
                flash("El correo ya está registrado", 'danger')
                return redirect(url_for('verSupervisor'))
        
        with mysql.connection.cursor() as regSupervisor:
            regSupervisor.execute("INSERT INTO supervisor (nombreSup, apellidoPSup, apellidoMSup, correoSup, contraSup, codVeriSup, activoSup, veriSup, priviSup) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (nombreSupCC, apellidoPSupCC, apellidoMSupCC, correoSup, hashed_password, codVeriSup, activoSup, veriSup, priviSup))
            mysql.connection.commit()

            # MANDAR CORREO CON CODIGO DE VERIRIFICACION
            idSup               = regSupervisor.lastrowid
            
        with mysql.connection.cursor() as selSup:
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
@login_required
@verified_required
@admin_required
def verSupervisor():
    # SE MANDA A LLAMAR LA FUNCION PARA DESENCRIPTAR
    encriptar = encriptado()

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['supervisor', 'activoSup']
    list_campo = ['nombreSup', 'apellidoPSup', 'apellidoMSup']
    
    # SE OBTIENEN LOS DATOS FORMATEADOS DE LA BASE DE DATOS PARA ENVIAR AL FRONT
    sup, datosSup = obtener_datos(list_campo, list_consult, mysql, encriptar, 1)

    return render_template('/adm/adm_super.html', super = sup, datosSup = datosSup, username=session['name'], email=session['correoAd'])

    
#~~~~~~~~~~~~~~~~~~~ Editar Supervisores ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/EditarCuentaSupervisor', methods=["GET", "POST"])
@require_post
def editarCuentaSupervisor():
    # Se crea un diccionario de validaciones
    campos_validacion_sup = {
        'nombreSup':    {'max_length': 20, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,20}"},
        'apellidoPSup': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"},
        'apellidoMSup': {'max_length': 15, 'pattern': r"[a-zA-ZàáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð ,.'\-]{2,15}"}
    }
    # Se crea un diccionario de etiquetas sobre los campos
    etiquetas_sup = {
        'nombreSup': 'Nombre',
        'apellidoPSup': 'Apellido Paterno',
        'apellidoMSup': 'Apellido Materno'
    }
    
    errores = validar_campos(request, campos_validacion_sup, etiquetas_sup)

    if errores:
        for error in errores:
            flash(error)
        return redirect(url_for('verSupervisor'))
    
        
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
@require_post
def eliminarCuentaSupervisor():
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = ['idSup', 'supervisor', 'activoSup']
    eliminarCuenta(request, mysql, list_consult)

    flash('Cuenta editada con exito.')
    return redirect(url_for('verSupervisor'))


@PCapp.route('/IndexAdministrador', methods=['GET', 'POST'])
@login_required
@verified_required
@admin_required
def indexAdministrador():
    return render_template('/adm/index_admin.html', username=session['name'], email=session['correoAd'], request=request)
    

#~~~~~~~~~~~~~~~~~~~ Index Practicantes ~~~~~~~~~~~~~~~~~~~#
@PCapp.route('/IndexPracticantes', methods=["GET", "POST"])
@login_required
@verified_required
@practicante_required
def indexPracticantes():
    #SE MANDA A LLAMRA LA FUNCION PARA ENCRIPTAR
    encriptar = encriptado()

    idPrac = session['idPrac']
    
    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_campo  = ['nombrePrac', 'apellidoPPrac', 'apellidoMPrac', 'nombrePaci', 'apellidoPPaci', 'apellidoMPaci']
    list_consult = [idPrac, 1]
    
    pra, datosPrac = obtener_datos(list_campo, list_consult, mysql, encriptar, 3)

    # SE CREAN LISTAS DE LOS DATOS REQUERIDOS
    list_consult = [idPrac, 4]
    praH, datosPracH = obtener_datos(list_campo, list_consult, mysql, encriptar, 3)

    return render_template('/prac/index_practicante.html', pract = pra, datosPrac = datosPrac, datosPracH=datosPracH, username=session['name'], email=session['correoPrac'], request=request)

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
@supervisor_required
def agregarPracticante():
    return render_template('/sup/agregar_practicante.html', username=session['name'], email=session['correoSup'])

@PCapp.route('/logout')
@login_required
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

@PCapp.cli.command()
def list_routes():
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        options = {}
        for arg in rule.arguments:
            options[arg] = "[{0}]".format(arg)

        methods = ','.join(rule.methods)
        url = url_for(rule.endpoint, **options)
        line = urllib.parse.unquote("{:50s} {:20s} {}".format(rule.endpoint, methods, url))
        output.append(line)

    for line in sorted(output):
        print(line)




if __name__ == '__main__':
    PCapp.secret_key = '123'
    PCapp.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=6) 
    csrf.init_app(PCapp)
    PCapp.register_error_handler(401,status_401)
    PCapp.register_error_handler(404,status_404)
    PCapp.run(port=3000,debug=True)
    
#Cuenta un chiste
# ¿Cuál es el animal más antiguo?
# La cebra, porque está en blanco y negro

#Cuenta otro chiste
# ¿Qué hace una abeja en el gimnasio?
# ¡Zum-ba!

#Uno mas
# ¿Por qué los pájaros no usan Facebook?
# Porque ya tienen Twitter.

# ¿Por qué los pájaros no usan computadoras?
# Porque les da miedo el ratón.


#E we, cuanto es 2 + 2?  = △⃒⃘

# FALTA PROBAR CITAS EN PACIENTES
# ELIMINAR CITAS (PACIENTES Y PRACTICANTES)
# ALL EL SISTEMA DE ENCUESTAS ZZZZZZZ