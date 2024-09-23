console.log('estoy cargado');
const surveyQuestions = [
    {
        question: "En las últimas dos semanas ¿con qué frecuencia te has sentido triste vacío o sin esperanza durante la mayor parte del día?",
        name: "depresion_sentirse_triste"
    },
    {
        question: "En las últimas dos semanas ¿en qué medida has perdido interés o placer en casi todas las actividades que solías disfrutar como pasatiempos o reuniones sociales?",
        name: "depresion_perdida_interes"
    },
    {
        question: "En las últimas dos semanas ¿has notado algún cambio significativo en tu peso (ya sea pérdida o aumento de peso sin estar a dieta) o en tu apetito?",
        name: "depresion_cambios_peso_apetito"
    },
    {
        question: "En las últimas dos semanas ¿cómo describirías tus patrones de sueño (por ejemplo dificultad para dormir dormir demasiado o interrumpido)?",
        name: "depresion_patrones_sueno"
    },
    {
        question: "En las últimas dos semanas ¿con qué frecuencia has experimentado fatiga o una pérdida de energía que afecta tu capacidad para realizar tareas diarias?",
        name: "depresion_fatiga_energia"
    },
    {
        question: "En las últimas semanas ¿con qué frecuencia te has sentido preocupado o inquieto por cosas que están pasando en tu vida diaria como el trabajo la escuela o la familia?",
        name: "ansiedad_preocupacion_diaria"
    },
    {
        question: "¿Te resulta difícil controlar tus preocupaciones incluso cuando intentas distraerte o relajarte?",
        name: "ansiedad_control_preocupacion"
    },
    {
        question: "En las últimas semanas ¿has sentido una sensación constante de estar en el borde o nervioso como si estuvieras en tensión todo el tiempo?",
        name: "ansiedad_sensacion_nerviosa"
    },
    {
        question: "¿Te has sentido más cansado de lo habitual incluso si no has estado haciendo mucho esfuerzo físico?",
        name: "ansiedad_fatiga_cansancio"
    },
    {
        question: "¿Te ha costado concentrarte en tus tareas diarias o en tus pensamientos porque sientes que tu mente está en blanco o te cuesta enfocarte?",
        name: "ansiedad_dificultad_concentracion"
    },
    {
        question: "En las últimas semanas ¿con qué frecuencia has sentido nervios o miedo intenso antes de tener que participar en actividades sociales como conocer a nuevas personas o hablar en público?",
        name: "ansiedad_miedo_social"
    },
    {
        question: "¿Te has preocupado mucho sobre la posibilidad de hacer el ridículo o de ser juzgado negativamente por otros cuando estás en situaciones sociales?",
        name: "ansiedad_preocupacion_evaluacion"
    },
    {
        question: "En las últimas semanas ¿has evitado eventos sociales o situaciones donde podría haber gente porque te sientes demasiado incómodo o ansioso?",
        name: "ansiedad_evitar_social"
    },
    {
        question: "¿Durante las interacciones sociales como charlar con amigos o familiares te has sentido tan incómodo o ansioso que te resulta difícil disfrutar del momento?",
        name: "ansiedad_malestar_interacciones"
    },
    {
        question: "¿Sientes que tu ansiedad en situaciones sociales afecta tu vida diaria como tu capacidad para hacer amigos tu desempeño en el trabajo o en la escuela o cómo te sientes en general?",
        name: "ansiedad_impacto_vida_diaria"
    },
    {
        question: "¿Te resulta complicado concentrarte en tareas o actividades durante un tiempo prolongado como leer un libro o trabajar en un proyecto y a menudo te das cuenta de que tu mente se desvía?",
        name: "tdah_dificultad_atencion"
    },
    {
        question: "¿Te pasa a menudo que olvidas cosas importantes como dónde dejaste las llaves o los deberes escolares y otros compromisos?",
        name: "tdah_olvidar_cosas"
    },
    {
        question: "¿Te cuesta trabajo organizar tus tareas y actividades como hacer una lista de cosas por hacer o mantenerte al día con tus responsabilidades?",
        name: "tdah_dificultad_organizacion"
    },
    {
        question: "¿Sueles interrumpir a los demás mientras están hablando o responder antes de que terminen de hacer una pregunta?",
        name: "tdah_impulsividad_hablar"
    },
    {
        question: "¿Te resulta difícil quedarte quieto o estar sentado en situaciones donde se espera que permanezcas tranquilo como en reuniones o en el cine?",
        name: "tdah_hiperactividad"
    },
    {
        question: "¿Te resulta fácil ignorar las reglas y normas ya sea en el trabajo en la escuela o en la vida cotidiana y hacer lo que te parece sin preocuparte mucho por las consecuencias para los demás?",
        name: "antisocial_normas"
    },
    {
        question: "¿Con qué frecuencia has encontrado que engañas o manipulas a otras personas para obtener lo que quieres incluso si eso significa mentir o usar excusas?",
        name: "antisocial_engaño_manipulacion"
    },
    {
        question: "¿A menudo tomas decisiones impulsivas sin pensar mucho en las consecuencias como gastar dinero sin planificación o actuar de manera arriesgada?",
        name: "antisocial_impulsividad"
    },
    {
        question: "¿Te resulta común tener conflictos o pelear con otras personas ya sea en casa en el trabajo o en otros lugares y sientes que te enojas fácilmente?",
        name: "antisocial_agresividad_conflictos"
    },
    {
        question: "Cuando haces algo que lastima a alguien o causa problemas ¿sientes que no te importa mucho o simplemente piensas que ellos deberían haberse hecho cargo de la situación?",
        name: "antisocial_falta_remordimientos"
    }
];



// Variable global para llevar el índice de la subcard actual
let currentSubcardIndex = 0;
const totalSubcards = 5; // Tenemos 5 páginas de preguntas

// Función para generar la encuesta
function generateSurvey() {
    const container = document.getElementById('survey-container');
    let subcardHtml = '';

    for (let i = 0; i < totalSubcards; i++) { // 5 páginas
        subcardHtml += `
        <div class="subcard mb-4 ${i === 0 ? 'activee' : ''}" id="subcard-${i}">
            <h5>Página ${i + 1}</h5>
            ${surveyQuestions.slice(i * 5, (i + 1) * 5).map((questionObj) => `
                <div class="custom-form-group">
                    <div class="justify-content-center form-check">
                        <label>${questionObj.question}</label>
                        <br>
                        <div class="custom-radio-container">
                            <!-- Opciones de radio -->
                            <label class="toggle mr-3">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    value="0"
                                    required><span class="label-text">Nunca</span>
                            </label>
                            <label class="toggle mr-3">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    value="1"
                                    required><span class="label-text">De vez en cuando</span>
                            </label>
                            <label class="toggle mr-3">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    value="2"
                                    required><span class="label-text">A menudo</span>
                            </label>
                            <label class="toggle mr-3">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    value="3"
                                    required><span class="label-text">Casi siempre</span>
                            </label>
                            <label class="toggle">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    value="4"
                                    required><span class="label-text">Siempre</span>
                            </label>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        `;
    }

    container.innerHTML = subcardHtml;
}

// Función para validar las preguntas de la página actual
function validatePage(currentPage) {
    const subcard = document.querySelector(`#subcard-${currentPage}`);
    let allAnswered = true;

    const questions = subcard.querySelectorAll('.custom-form-group');
    questions.forEach(question => {
        const radios = question.querySelectorAll('input[type="radio"]:checked');
        if (radios.length === 0) {
            allAnswered = false;
        }
    });

    return allAnswered;
}

// Función para validar que todas las preguntas han sido respondidas
function validateAllQuestions() {
    const subcards = document.querySelectorAll('.subcard');
    let allAnswered = true;

    subcards.forEach(subcard => {
        const questions = subcard.querySelectorAll('.custom-form-group');
        questions.forEach(question => {
            const radios = question.querySelectorAll('input[type="radio"]:checked');
            if (radios.length === 0) {
                allAnswered = false;
            }
        });
    });

    return allAnswered;
}

// Función para enviar el formulario
function submitForm() {
    // Mostrar la última tarjeta
    $('.card').removeClass('show').addClass('previous');
    $('.card:last-child').removeClass('previous').addClass('show');

    // Esperar 5 segundos antes de enviar el formulario
    setTimeout(() => {
        document.getElementById('surveyForm').submit();
    }, 5000);
}

// Función para desplazarse al inicio de la página
function scrollToTop() {
    document.body.scrollTop = 0; // Para Safari
    document.documentElement.scrollTop = 0; // Para Chrome, Firefox, IE y Opera
}

// Función para actualizar la barra de progreso
function updateProgress(currentPage, totalPages) {
    const progress = (currentPage / totalPages) * 100;
    document.querySelector('.progress').style.width = `${progress}%`;
}

// Función para mostrar la subcard (página de preguntas) específica
function showSubcard(index, direction) {
    const currentSubcard = $('.subcard.activee');
    const nextSubcard = $(`#subcard-${index}`);

    currentSubcard.removeClass('activee').addClass('previous');
    nextSubcard.removeClass('previous').addClass('activee');

    updateProgress(index + 1, totalSubcards);
    currentSubcardIndex = index;
    updateButtons();

    scrollToTop();
}

// Función para cambiar de tarjeta
function changeCard(current, next, direction) {
    current.removeClass('show').addClass('previous');
    next.removeClass('previous').addClass('show');

    // Asegurarse de que la tarjeta activa tenga el z-index más alto
    $('.card').css('z-index', 1);
    next.css('z-index', 2);
}

// Función para restablecer el z-index
function resetZIndex() {
    $('.card').css('z-index', 1);
    $('.card:first-child').css('z-index', 2);
}

// Función para actualizar los botones
function updateButtons() {
    const currentCardIndex = $('.card').index($('.card.show'));

    if (currentCardIndex === 0) {
        $('.prev').hide();
        $('#next1').show();
        $('#next2, #next3').hide();
        $('.progress-container').hide(); // Ocultar barra de progreso en la primera tarjeta
    } else if (currentCardIndex === 1) {
        $('.prev').show();
        $('#next1').hide();
        $('#next2').show().text(currentSubcardIndex === totalSubcards - 1 ? 'CONFIRMAR' : 'SIGUIENTE');
        $('#next3').hide();
        $('.progress-container').show(); // Mostrar barra de progreso en las páginas de preguntas
    } else if (currentCardIndex === 2) {
        $('.prev').show();
        $('#next1, #next2').hide();
        $('#next3').show();
        $('.progress-container').hide(); // Ocultar barra de progreso en la página de confirmación
        scrollToTop(); // Llamar a scrollToTop al llegar a la página de confirmación
    } else {
        $('.prev, #next1, #next2, #next3').hide();
        $('.progress-container').hide(); // Ocultar barra de progreso en otras tarjetas
    }
}

// Document Ready
$(document).ready(function () {
    console.log('ready');
    generateSurvey();
    updateButtons();
    resetZIndex();

    var current_fs, next_fs, previous_fs;

    // Inicializar la barra de progreso
    updateProgress(1, totalSubcards);

    // Evento al hacer clic en los botones "next"
    $(".next").click(function () {
        var id = $(this).attr('id');

        if (id === "next1" && document.getElementById("customCheck1").checked) {
            current_fs = $(this).closest('.card');
            next_fs = current_fs.next();
            changeCard(current_fs, next_fs, 'next');
            showSubcard(0, 'next');
        } else if (id === "next2") {
            if (!validatePage(currentSubcardIndex)) {
                alert("Por favor, responda todas las preguntas antes de continuar.");
                return;
            }
            if (currentSubcardIndex < totalSubcards - 1) {
                showSubcard(currentSubcardIndex + 1, 'next');
            } else {
                current_fs = $(this).closest('.card');
                next_fs = current_fs.next();
                changeCard(current_fs, next_fs, 'next');
            }
        } else if (id === "next3") {
            if (validateAllQuestions()) {
                submitForm();
            } else {
                alert("Por favor, responda todas las preguntas antes de continuar.");
            }
        }

        updateButtons();
    });

    // Evento al hacer clic en los botones "prev"
    $(".prev").click(function () {
        current_fs = $(this).closest('.card');
        var currentCardIndex = $('.card').index(current_fs);

        if (currentCardIndex === 2) { // Si estamos en la card de CONFIRMACION
            previous_fs = current_fs.prev();
            changeCard(current_fs, previous_fs, 'prev');
            showSubcard(totalSubcards - 1, 'prev'); // Mostrar la última subcard
        } else if (currentSubcardIndex > 0) {
            showSubcard(currentSubcardIndex - 1, 'prev');
        } else {
            previous_fs = current_fs.prev();
            changeCard(current_fs, previous_fs, 'prev');
            if (previous_fs.index() === 0) {
                resetZIndex();
            }
        }

        updateButtons();
    });

    // Evento al hacer clic en el botón "next3"
    $("#next3").click(function () {
        if (validateAllQuestions()) {
            submitForm();
        } else {
            alert("Por favor, responda todas las preguntas antes de continuar.");
        }
    });
});