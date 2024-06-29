from flask                  import Flask, render_template, request, redirect, url_for, session, flash, make_response
from flask_mysqldb          import MySQL, MySQLdb
from typing                 import List, Dict

app     = Flask(__name__)
mysql   = MySQL(app)
app.config['MYSQL_HOST']              = 'localhost'
app.config['MYSQL_USER']              = 'root'
app.config['MYSQL_PASSWORD']          = 'root'
app.config['MYSQL_DB']                = 'prueba'



# Define symptoms for each disorder
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    reported_symptoms = [symptom for symptom, value in request.form.items() if value == 'si']
    print(reported_symptoms)
    diagnoses_p , diagnoses_s = diagnose(reported_symptoms)
    print(diagnoses_p)
    print(diagnoses_s)
    cur = mysql.connection.cursor()
    nombre = request.form['nombre']
    correo = request.form['correo']
    
    
    cur.execute("INSERT INTO usuarios(nombre, correo,respuestas, sint_pri, sint_sec) VALUES(%s,%s,%s,%s,%s)",
                 (nombre, correo, ','.join(reported_symptoms), diagnoses_p, diagnoses_s))
    mysql.connection.commit()
    
    cur.execute(" SELECT * FROM usuarios")
    pacientes = []
    column_names = [desc[0] for desc in cur.description]
    
    for row in cur.fetchall():
        pacientes.append(dict(zip(column_names, row)))

    cur.close()
    return render_template('resultado.html', diagnoses_p=diagnoses_p, diagnoses_s=diagnoses_s, pacientes=pacientes)


if __name__ == '__main__':
    app.run(debug=True)