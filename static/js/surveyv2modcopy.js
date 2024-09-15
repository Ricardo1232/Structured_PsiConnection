console.log('estoy cargado');
const surveyQuestions = [
    {
        question: "¿Te sientes nervioso o ansioso con frecuencia sin razón aparente?",
        name: "ansiedad_nervioso_ansioso"
    },
    {
        question: "¿Experimentas síntomas físicos como palpitaciones o sudoración cuando estás estresado?",
        name: "ansiedad_sintomas_fisicos"
    },
    {
        question: "¿Tiendes a preocuparte excesivamente por situaciones futuras?",
        name: "ansiedad_preocupacion_excesiva"
    },
    {
        question: "¿Evitas situaciones sociales por miedo a sentirte incómodo o juzgado?",
        name: "ansiedad_evitar_social"
    },
    {
        question: "¿Tienes dificultades para conciliar el sueño debido a preocupaciones?",
        name: "ansiedad_problemas_sueno"
    },
    {
        question: "¿Has perdido interés en actividades que antes disfrutabas?",
        name: "depresion_perdida_interes"
    },
    {
        question: "¿Te sientes triste o desanimado la mayor parte del día?",
        name: "depresion_tristeza"
    },
    {
        question: "¿Has experimentado cambios significativos en tus patrones de sueño o apetito?",
        name: "depresion_cambios_sueno_apetito"
    },
    {
        question: "¿Te sientes sin energía o fatigado con frecuencia?",
        name: "depresion_fatiga"
    },
    {
        question: "¿Tienes pensamientos de inutilidad o culpa excesiva?",
        name: "depresion_inutilidad_culpa"
    },
    {
        question: "¿Te cuesta respetar las normas sociales o legales?",
        name: "antisocial_respetar_normas"
    },
    {
        question: "¿Tienes dificultades para sentir empatía por los demás?",
        name: "antisocial_falta_empatia"
    },
    {
        question: "¿Actúas de manera impulsiva o irresponsable con frecuencia?",
        name: "antisocial_impulsivo_irresponsable"
    },
    {
        question: "¿Sientes indiferencia por los sentimientos o derechos de los demás?",
        name: "antisocial_indiferencia_derechos"
    },
    {
        question: "¿Tiendes a manipular a otros para obtener beneficios personales?",
        name: "antisocial_manipulacion"
    },
    {
        question: "¿Te resulta difícil mantener la atención en clases o durante el estudio?",
        name: "tdah_atencion"
    },
    {
        question: "¿Sueles perder objetos necesarios para tus actividades (como llaves, libros, etc.)?",
        name: "tdah_perder_objetos"
    },
    {
        question: "¿Te sientes inquieto o tienes dificultades para permanecer sentado por largos períodos?",
        name: "tdah_inquietud"
    },
    {
        question: "¿Interrumpes a otros o hablas en momentos inapropiados?",
        name: "tdah_interrupcion"
    },
    {
        question: "¿Tienes problemas para organizar tareas y actividades?",
        name: "tdah_organizacion"
    },
    {
        question: "¿Experimentas períodos de energía excesiva y menor necesidad de dormir?",
        name: "bipolar_energia_excesiva"
    },
    {
        question: "¿Has tenido episodios de tristeza profunda alternados con períodos de euforia?",
        name: "bipolar_cambios_estado_animo"
    },
    {
        question: "¿Notas cambios significativos en tu autoestima, pasando de muy alta a muy baja?",
        name: "bipolar_cambios_autoestima"
    },
    {
        question: "¿Has tenido episodios de gastos excesivos o comportamientos impulsivos?",
        name: "bipolar_gastos_impulsivos"
    },
    {
        question: "¿Experimentas cambios rápidos en tus ideas o planes para el futuro?",
        name: "bipolar_cambios_rapidos_ideas"
    },
    {
        question: "¿Tienes pensamientos intrusivos y recurrentes que te causan ansiedad?",
        name: "toc_pensamientos_intrusivos"
    },
    {
        question: "¿Realizas rituales o acciones repetitivas para aliviar la ansiedad?",
        name: "toc_rituales"
    },
    {
        question: "¿Dedicas mucho tiempo a estas obsesiones o compulsiones, interfiriendo con tu vida diaria?",
        name: "toc_interferencia_diaria"
    },
    {
        question: "¿Sientes la necesidad de comprobar las cosas repetidamente?",
        name: "toc_comprobacion_repetida"
    },
    {
        question: "¿Te resulta difícil controlar estos pensamientos o comportamientos?",
        name: "toc_dificultad_control"
    },
    {
        question: "¿Ciertos sonidos cotidianos (como masticar, respirar fuerte) te provocan una reacción emocional intensa?",
        name: "misofonia_sensibilidad_sonidos"
    },
    {
        question: "¿Evitas situaciones sociales debido a tu sensibilidad a ciertos sonidos?",
        name: "misofonia_evitar_situaciones"
    },
    {
        question: "¿Tu reacción a estos sonidos interfiere con tu capacidad para concentrarte en tus estudios?",
        name: "misofonia_interferencia_concentracion"
    },
    {
        question: "¿Sientes ira o disgusto intenso cuando escuchas estos sonidos?",
        name: "misofonia_ira_disgusto"
    },
    {
        question: "¿Has notado que tu sensibilidad a ciertos sonidos ha aumentado con el tiempo?",
        name: "misofonia_aumento_sensibilidad"
    }
];


function generateSurvey() {
    const container = document.getElementById('survey-container');
    const form = document.getElementById('surveyForm');
    let subcardHtml = '';

    for (let i = 0; i < 9; i++) {
        subcardHtml += `
        <div class="subcard mb-4 ${i === 0 ? 'activee' : ''}" id="subcard-${i}">
            <h5>Pagina ${i + 1}</h5>
            ${surveyQuestions.slice(i * 5, (i + 1) * 5).map((questionObj, index) => `
                <div class="form-group">
                    <div class="justify-content-center form-check">
                        <label>${questionObj.question}</label>
                        <br>
                        <div class="radio-container d-inline">
                            <label class="toggle mr-3">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    id="${questionObj.name}Si"
                                    value="2"
                                    required><span
                                    class="label-text">Sí</span>
                            </label>
                            <label class="toggle">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    id="${questionObj.name}Aveces"
                                    value="1"
                                    required><span
                                    class="label-text">A veces</span>
                            </label>
                            <label class="toggle">
                                <input
                                    class="form-check-input"
                                    type="radio"
                                    name="${questionObj.name}"
                                    id="${questionObj.name}No"
                                    value="0"
                                    required><span
                                    class="label-text">No</span>
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

// Call this function when the page loads
document.addEventListener('DOMContentLoaded', generateSurvey);

function validateSurvey() {
    const subcards = document.querySelectorAll('.subcard');
    let allAnswered = true;

    subcards.forEach(subcard => {
        const questions = subcard.querySelectorAll('.form-group');
        questions.forEach(question => {
            const radios = question.querySelectorAll('input[type="radio"]:checked');
            if (radios.length === 0) {
                allAnswered = false;
            }
        });
    });

    return allAnswered;
}

function submitForm() {
    if (validateSurvey()) {
        // Mostrar la última tarjeta
        $('.card').removeClass('show').addClass('previous');
        $('.card:last-child').removeClass('previous').addClass('show');

        // Esperar 5 segundos antes de enviar el formulario
        setTimeout(() => {
            document.getElementById('surveyForm').submit();
        }, 5000);
    } else {
        alert("Por favor, responda todas las preguntas antes de continuar.");
    }
}

$(document).ready(function () {
    console.log('ready');
    generateSurvey();
    updateButtons();
    resetZIndex();
    var current_fs, next_fs, previous_fs;
    var currentSubcardIndex = 0;
    var totalSubcards = 7;

    function showSubcard(index, direction) {
        var currentSubcard = $('.subcard.activee');
        var nextSubcard = $(`#subcard-${index}`);

        if (direction === 'next') {
            currentSubcard.removeClass('activee').addClass('previous');
            nextSubcard.removeClass('previous').addClass('activee');
        } else {
            currentSubcard.removeClass('activee');
            nextSubcard.removeClass('previous').addClass('activee');
        }

        currentSubcardIndex = index;
        updateButtons();
    }

    function updateButtons() {
        if ($('.card.show').index() === 0) {
            $('.prev').hide();
            $('#next1').show();
            $('#next2, #next3').hide();
        } else if ($('.card.show').index() === 1) {
            $('.prev').show();
            $('#next1').hide();
            $('#next2').show().text(currentSubcardIndex === totalSubcards - 1 ? 'CONFIRMAR' : 'SIGUIENTE');
            $('#next3').hide();
        } else if ($('.card.show').index() === 2) {
            $('.prev').show();
            $('#next1, #next2').hide();
            $('#next3').show();
        } else {
            $('.prev, #next1, #next2, #next3').hide();
        }
    }

    function changeCard(current, next, direction) {
        if (direction === 'next') {
            current.removeClass('show').addClass('previous');
            next.removeClass('previous').addClass('show');
        } else {
            next.removeClass('previous').addClass('show');
            current.removeClass('show').addClass('previous');
        }

        // Asegurarse de que la tarjeta activa tenga el z-index más alto
        $('.card').css('z-index', 1);
        next.css('z-index', 2);
    }

    function resetZIndex() {
        $('.card').css('z-index', 1);
        $('.card:first-child').css('z-index', 2);
    }

    $(".next").click(function () {
        var id = $(this).attr('id');

        if (id === "next1" && document.getElementById("customCheck1").checked) {
            current_fs = $(this).closest('.card');
            next_fs = current_fs.next();
            changeCard(current_fs, next_fs, 'next');
            showSubcard(0, 'next');
        } else if (id === "next2") {
            if (currentSubcardIndex < totalSubcards - 1) {
                showSubcard(currentSubcardIndex + 1, 'next');
            } else if (validateSurvey()) {
                current_fs = $(this).closest('.card');
                next_fs = current_fs.next();
                changeCard(current_fs, next_fs, 'next');
            } else {
                alert("Por favor, responda todas las preguntas antes de continuar.");
            }
        } else if (id === "next3") {
            if (validateSurvey()) {
                submitForm();
            } else {
                alert("Por favor, responda todas las preguntas antes de continuar.");
            }
        }

        updateButtons();
    });


    $(".prev").click(function () {
        current_fs = $(this).closest('.card');
        var currentCardIndex = $('.card').index(current_fs);

        if (currentCardIndex === 2) { // Si estamos en la card de CONFIRMACION
            previous_fs = current_fs.prev();
            changeCard(current_fs, previous_fs, 'prev');
            showSubcard(totalSubcards - 1, 'prev'); // Mostrar la última subcard de la encuesta
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

    $("#next3").click(function () {
        submitForm();
    });

    // Inicializar los botones
    updateButtons();
});

function validateSurvey() {
    const subcards = document.querySelectorAll('.subcard');
    let allAnswered = true;

    subcards.forEach(subcard => {
        const questions = subcard.querySelectorAll('.form-group');
        questions.forEach(question => {
            const radios = question.querySelectorAll('input[type="radio"]:checked');
            if (radios.length === 0) {
                allAnswered = false;
            }
        });
    });

    return allAnswered;
}




