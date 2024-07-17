if (currentUrl == "SurveyV2modcopy"){
    console.log("PEDRO PEDRO PEDRO PE   ")
    
    const surveyQuestions = [
        {
            question: "¿Siente una preocupación excesiva sobre diferentes aspectos de su vida?",
            name: "preocupacion_excesiva"
        },
        {
            question: "¿Se siente nervioso o está constantemente en alerta?",
            name: "nerviosismo"
        },
        {
            question: "¿Se siente fatigado la mayor parte del tiempo?",
            name: "fatiga"
        },
        {
            question: "¿Tiene problemas de concentración?",
            name: "problemas_concentracion"
        },
        {
            question: "¿Se irrita fácilmente?",
            name: "irritabilidad"
        },
        {
            question: "¿Siente tensión muscular?",
            name: "tension_muscular"
        },
        {
            question: "¿Tiene problemas para conciliar el sueño?",
            name: "problemas_sueno"
        },
        {
            question: "¿Experimenta sentimientos de tristeza profunda?",
            name: "sentimientos_tristeza"
        },
        {
            question: "¿Ha perdido interés en actividades que solía disfrutar?",
            name: "perdida_interes"
        },
        {
            question: "¿Ha experimentado cambios en su apetito o peso recientemente?",
            name: "cambios_apetito_peso"
        },
        {
            question: "¿Ha tenido pensamientos de suicidio?",
            name: "pensamientos_suicidio"
        },
        {
            question: "¿Tiene dificultades para mantener la atención en una tarea?",
            name: "dificultad_atencion"
        },
        {
            question: "¿Se considera una persona hiperactiva?",
            name: "hiperactividad"
        },
        {
            question: "¿Suele actuar impulsivamente?",
            name: "impulsividad"
        },
        {
            question: "¿Encuentra dificultades para seguir instrucciones?",
            name: "dificultades_instrucciones"
        },
        {
            question: "¿Experimenta temblores cuando está en reposo?",
            name: "temblor_reposo"
        },
        {
            question: "¿Siente rigidez muscular?",
            name: "rigidez_muscular"
        },
        {
            question: "¿Ha notado lentitud en sus movimientos?",
            name: "lentitud_movimientos"
        },
        {
            question: "¿Tiene problemas de equilibrio o coordinación?",
            name: "problemas_equilibrio_coordinacion"
        },
        {
            question: "¿Tiene dificultades para hablar o escribir?",
            name: "dificultad_hablar_escribir"
        },
        {
            question: "¿Ha experimentado pérdida de memoria recientemente?",
            name: "perdida_memoria"
        },
        {
            question: "¿Encuentra dificultades para encontrar palabras en conversaciones?",
            name: "dificultad_palabras_conversaciones"
        },
        {
            question: "¿Se siente desorientado(a) en tiempo o espacio?",
            name: "desorientacion_espacial_temporal"
        },
        {
            question: "¿Ha experimentado cambios en su estado de ánimo o comportamiento?",
            name: "cambios_estado_animo_comportamiento"
        },
        {
            question: "¿Encuentra dificultades para realizar tareas cotidianas?",
            name: "dificultad_tareas_cotidianas"
        },
        {
            question: "¿Experimenta episodios de manía?",
            name: "episodios_mania"
        },
        {
            question: "¿Experimenta episodios de depresión?",
            name: "episodios_depresion"
        },
        {
            question: "¿Experimenta cambios bruscos en su humor o actividad?",
            name: "cambios_bruscos_humor_actividad"
        },
        {
            question: "¿Experimenta obsesiones recurrentes?",
            name: "obsesiones"
        },
        {
            question: "¿Siente la necesidad de realizar compulsiones para aliviar la ansiedad?",
            name: "compulsiones"
        },
        {
            question: "¿Reconoce la ineficacia de sus intentos por controlar sus pensamientos o comportamientos obsesivos?",
            name: "reconocimiento_ineficacia_control"
        },
        {
            question: "¿Se siente irritado(a) ante ciertos sonidos?",
            name: "irritabilidad_misofonia"
        },
        {
            question: "¿Se enfada fácilmente debido a sonidos específicos?",
            name: "enfado"
        },
        {
            question: "¿Experimenta ansiedad intensa ante ciertos sonidos?",
            name: "ansiedad"
        },
        {
            question: "¿Siente náuseas ante ciertos sonidos?",
            name: "nauseas"
        },
        {
            question: "¿Experimenta sudoración excesiva ante ciertos sonidos?",
            name: "sudoracion"
        },
        {
            question: "¿Siente la necesidad de escapar de ciertos sonidos?",
            name: "necesidad_escapar"
        },
        {
            question: "¿Ciertos sonidos desencadenan reacciones negativas en usted?",
            name: "sonidos_desencadenantes"
        },
        {
            question: "¿Ha violado los derechos de los demás, como engañar, robar o agredir físicamente?",
            name: "violacion_derechos"
        },
        {
            question: "¿Desprecia o viola las normas sociales y legales, mostrando indiferencia por la seguridad propia o de los demás?",
            name: "desprecio_normas"
        },
        {
            question: "¿Muestra irresponsabilidad consistente, como no cumplir con obligaciones laborales o financieras?",
            name: "irresponsabilidad"
        },
        {
            question: "¿Presenta falta de remordimiento, indiferencia o justificación de haber dañado, maltratado o robado a otras personas?",
            name: "falta_remordimiento"
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
                                        value="si"
                                        required><span
                                        class="label-text">Sí</span>
                                </label>
                                <label class="toggle">
                                    <input
                                        class="form-check-input"
                                        type="radio"
                                        name="${questionObj.name}"
                                        id="${questionObj.name}No"
                                        value="no"
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

    $(document).ready(function(){
        generateSurvey();
        updateButtons();
        resetZIndex();
        var current_fs, next_fs, previous_fs;
        var currentSubcardIndex = 0;
        var totalSubcards = 9;

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

        $(".next").click(function(){
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


        $(".prev").click(function(){
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

        $("#next3").click(function(){
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





}