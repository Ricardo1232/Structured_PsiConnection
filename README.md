# PsiConnection

PsiConnection es una aplicación web desarrollada en Python utilizando el framework Flask. Está diseñada para gestionar un sistema de conexión entre pacientes, practicantes, supervisores y administradores en el ámbito de la psicología clínica. La plataforma permite la gestión de usuarios, programación de citas, realización de encuestas y seguimiento de diagnósticos.

## Tabla de Contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación y Configuración](#instalación-y-configuración)
  - [Requisitos Previos](#requisitos-previos)
  - [Pasos de Instalación](#pasos-de-instalación)
- [Uso](#uso)
- [Consideraciones de Seguridad](#consideraciones-de-seguridad)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Características

- **Autenticación y Registro de Usuarios**: Permite a pacientes, practicantes, supervisores y administradores registrarse y autenticarse en la plataforma.
- **Gestión de Citas**: Los pacientes pueden programar citas con practicantes disponibles, integrando Google Calendar API para gestionar los eventos.
- **Encuestas y Diagnósticos**: Los pacientes completan encuestas iniciales cuyos resultados son procesados y almacenados para análisis posterior.
- **Roles y Permisos**: Control de acceso basado en roles, asegurando que cada tipo de usuario tenga acceso únicamente a las funcionalidades correspondientes.
- **Encriptación de Datos Sensibles**: Utiliza criptografía para encriptar información sensible antes de almacenarla en la base de datos.
- **Notificaciones por Correo Electrónico**: Envío de correos electrónicos para verificación de cuentas y notificaciones de citas.

## Arquitectura

La aplicación sigue una arquitectura **Modelo-Vista-Controlador (MVC)**:

- **Modelo (Model)**: Representado por las tablas de la base de datos MySQL y las operaciones CRUD asociadas.
- **Vista (View)**: Plantillas HTML renderizadas utilizando Jinja2, que presentan la información al usuario.
- **Controlador (Controller)**: Funciones de ruta que manejan la lógica de negocio, validaciones y comunicación entre el modelo y la vista.

## Tecnologías Utilizadas

- **Backend**:
  - Python 3.x
  - Flask
  - Flask-MySQLdb
  - Flask-Mail
  - Flask-Bcrypt
  - Flask-WTF
  - Cryptography (Fernet)
  - Google Calendar API
- **Base de Datos**:
  - MySQL
- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Bootstrap (opcional)
- **Otros**:
  - Jinja2 Templates
  - Werkzeug
  - PDFKit
  - Numpy
  - PyTorch (comentado para futuras implementaciones de IA)

## Instalación y Configuración

### Requisitos Previos

- Python 3.x instalado en el sistema.
- MySQL Server instalado y configurado.
- Acceso a una cuenta de Gmail para el envío de correos electrónicos (se recomienda crear una cuenta específica para este propósito).
- Credenciales para Google Calendar API.

### Pasos de Instalación

1. **Clonar el Repositorio**

   ```bash
   git clone https://github.com/tu_usuario/psiconnection.git
   cd psiconnection
   ```

2. **Crear y Activar un Entorno Virtual**

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar las Dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar la Base de Datos**

   - Crear una base de datos en MySQL llamada `psiconnection`.
   - Importar las tablas necesarias utilizando un script SQL o herramientas como phpMyAdmin.
   - Actualizar las credenciales de la base de datos en la configuración de la aplicación:

     ```python
     PCapp.config['MYSQL_HOST'] = 'localhost'
     PCapp.config['MYSQL_USER'] = 'tu_usuario'
     PCapp.config['MYSQL_PASSWORD'] = 'tu_contraseña'
     PCapp.config['MYSQL_DB'] = 'psiconnection'
     ```

5. **Configurar el Correo Electrónico**

   - Habilitar el acceso de aplicaciones menos seguras en la cuenta de Gmail (no recomendado para cuentas personales).
   - Generar una contraseña de aplicación si se utiliza autenticación de dos factores.
   - Actualizar la configuración de correo en la aplicación:

     ```python
     PCapp.config['MAIL_SERVER'] = 'smtp.gmail.com'
     PCapp.config['MAIL_PORT'] = 465
     PCapp.config['MAIL_USERNAME'] = 'tu_correo@gmail.com'
     PCapp.config['MAIL_PASSWORD'] = 'tu_contraseña_de_aplicación'
     PCapp.config['MAIL_USE_TLS'] = False
     PCapp.config['MAIL_USE_SSL'] = True
     ```

6. **Configurar Google Calendar API**

   - Crear un proyecto en Google Cloud Platform y habilitar la API de Google Calendar.
   - Descargar el archivo `credentials.json` y colocarlo en el directorio raíz de la aplicación.
   - Asegurarse de que el alcance (SCOPES) esté configurado:

     ```python
     SCOPES = ['https://www.googleapis.com/auth/calendar']
     ```

7. **Ejecutar la Aplicación**

   ```bash
   flask run
   ```

   La aplicación estará disponible en [http://localhost:5000](http://localhost:5000).

## Uso

- **Registro y Autenticación**: Los usuarios pueden registrarse según su rol y deben verificar su cuenta mediante un código enviado por correo electrónico.
- **Programación de Citas**: Los pacientes pueden ver los horarios disponibles de los practicantes y programar citas.
- **Realización de Encuestas**: Los pacientes completan una encuesta inicial que ayuda en el diagnóstico.
- **Gestión de Usuarios**: Los administradores pueden crear, editar y eliminar cuentas de supervisores y practicantes.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT.
