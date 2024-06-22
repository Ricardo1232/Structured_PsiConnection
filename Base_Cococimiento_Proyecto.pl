%Sintomas_del_trastorno_de_ansiedad
sintoma(preocupacion_excesiva).
sintoma(nerviosismo).
sintoma(fatiga).
sintoma(problemas_concentracion).
sintoma(irritabilidad).
sintoma(tension_muscular).
sintoma(problemas_sueno).
sintoma(sentimientos_tristeza).
sintoma(perdida_interes).
sintoma(cambios_apetito_peso).
sintoma(pensamientos_suicidio).
sintoma(dificultad_atencion).
sintoma(hiperactividad).
sintoma(impulsividad).
sintoma(dificultades_instrucciones).
sintoma(temblor_reposo).
sintoma(rigidez_muscular).
sintoma(lentitud_movimientos).
sintoma(problemas_equilibrio_coordinacion).
sintoma(dificultad_hablar_escribir).
sintoma(perdida_memoria).
sintoma(dificultad_palabras_conversaciones).
sintoma(desorientacion_espacial_temporal).
sintoma(cambios_estado_animo_comportamiento).
sintoma(dificultad_tareas_cotidianas).
sintoma(episodios_mania).
sintoma(episodios_depresion).
sintoma(cambios_bruscos_humor_actividad).
sintoma(obsesiones).
sintoma(compulsiones).
sintoma(reconocimiento_ineficacia_control).
sintoma(irritabilidad).
sintoma(enfado).
sintoma(ansiedad).
sintoma(nauseas).
sintoma(sudoracion).
sintoma(necesidad_escapar).
sintoma(sonidos_desencadenantes).

%Preguntas_para_el_diagnstico
%PREGUNTAS_TRANSTORNO_DE_ANSIEDAD
pregunta(preocupacion_excesiva) :- write('�Siente una preocupaci�n excesiva sobre diferentes aspectos de su vida? (si/no)'), nl.
pregunta(nerviosismo):-write('�Se siente nervioso o est� constantemente en alerta? (si/no)'), nl.
pregunta(fatiga) :- write('�Se siente fatigado la mayor parte del tiempo? (si/no)'), nl.
pregunta(problemas_concentracion):- write('�Tiene problemas de concentraci�n? (s�/no)'), nl.
pregunta(irritabilidad):- write('�Se irrita f�cilmente? (s�/no)'), nl.
pregunta(tension_muscular):- write('�Siente tensi�n muscular? (s�/no)'), nl.
pregunta(problemas_sueno):- write('�Tiene problemas para conciliar el sue�o? (s�/no)'), nl.

%PREGUNTASDEPRESION
pregunta(sentimientos_tristeza):- write('�Experimenta sentimientos de tristeza profunda? (s�/no)'), nl.
pregunta(perdida_interes):- write( '�Ha perdido inter�s en actividades que sol�a disfrutar? (s�/no)'), nl.
pregunta(cambios_apetito_peso):- write( '�Ha experimentado cambios en su apetito o peso recientemente? (s�/no)'), nl.
pregunta(pensamientos_suicidio):- write( '�Ha tenido pensamientos de suicidio? (s�/no)'), nl.

%PREGUNTASTDAH
pregunta(dificultad_atencion):- write( '�Tiene dificultades para mantener la atenci�n en una tarea? (s�/no)'), nl.
pregunta(hiperactividad):- write( '�Se considera una persona hiperactiva? (s�/no)'), nl.
pregunta(impulsividad):- write( '�Suele actuar impulsivamente? (s�/no)'), nl.
pregunta(dificultades_instrucciones):- write( '�Encuentra dificultades para seguir instrucciones? (s�/no)'), nl.

%PREGUNTASPARKINSON
pregunta(temblor_reposo):- write( '�Experimenta temblores cuando est� en reposo? (s�/no)'), nl.
pregunta(rigidez_muscular):- write( '�Siente rigidez muscular? (s�/no)'), nl.
pregunta(lentitud_movimientos):- write( '�Ha notado lentitud en sus movimientos? (s�/no)'), nl.
pregunta(problemas_equilibrio_coordinacion):- write( '�Tiene problemas de equilibrio o coordinaci�n? (s�/no)'), nl.
pregunta(dificultad_hablar_escribir):- write( '�Tiene dificultades para hablar o escribir? (s�/no)'), nl.

%PREGUNTASalzheimer
pregunta(perdida_memoria):- write( '�Ha experimentado p�rdida de memoria recientemente? (s�/no)'), nl.
pregunta(dificultad_palabras_conversaciones):- write( '�Encuentra dificultades para encontrar palabras en conversaciones? (s�/no)'), nl.
pregunta(desorientacion_espacial_temporal):- write( '�Se siente desorientado(a) en tiempo o espacio? (s�/no)'), nl.
pregunta(cambios_estado_animo_comportamiento):- write( '�Ha experimentado cambios en su estado de �nimo o comportamiento? (s�/no)'), nl.
pregunta(dificultad_tareas_cotidianas):- write( '�Encuentra dificultades para realizar tareas cotidianas? (s�/no)'), nl.

%PREGUNTASBIPOLAR
pregunta(episodios_mania):- write( '�Experimenta episodios de man�a? (s�/no)'), nl.
pregunta(episodios_depresion):- write( '�Experimenta episodios de depresi�n? (s�/no)'), nl.
pregunta(cambios_bruscos_humor_actividad):- write( '�Experimenta cambios bruscos en su humor o actividad? (s�/no)'), nl.

%PREGUNTASTOC
pregunta(obsesiones):- write( '�Experimenta obsesiones recurrentes? (s�/no)'), nl.
pregunta(compulsiones):- write( '�Siente la necesidad de realizar compulsiones para aliviar la ansiedad? (s�/no)'), nl.
pregunta(reconocimiento_ineficacia_control):- write( '�Reconoce la ineficacia de sus intentos por controlar sus pensamientos o comportamientos obsesivos? (s�/no)'), nl.

%PREGUNTASMISOFONIA
pregunta(irritabilidad):- write( '�Se siente irritado(a) ante ciertos sonidos? (s�/no)'), nl.
pregunta(enfado):- write( '�Se enfada f�cilmente debido a sonidos espec�ficos? (s�/no)'), nl.
pregunta(ansiedad):- write( '�Experimenta ansiedad intensa ante ciertos sonidos? (s�/no)'), nl.
pregunta(nauseas):- write( '�Siente n�useas ante ciertos sonidos? (s�/no)'), nl.
pregunta(sudoracion):- write( '�Experimenta sudoraci�n excesiva ante ciertos sonidos? (s�/no)'), nl.
pregunta(necesidad_escapar):- write( '�Siente la necesidad de escapar de ciertos sonidos? (s�/no)'), nl.
pregunta(sonidos_desencadenantes):- write( '�Ciertos sonidos desencadenan reacciones negativas en usted? (s�/no)'), nl.

%PREGUNTASANTISOCIAL
pregunta(desprecio_normas_sociales):- write('�Siente desprecio o falta de respeto por las normas sociales? (s�/no)'), nl.
pregunta(manipulacion_engano):- write('�Suele manipular o enga�ar a los dem�s para obtener lo que quiere? (s�/no)'), nl.
pregunta(falta_empatia_remordimiento):- write('�Le resulta dif�cil sentir empat�a o remordimiento por las acciones que ha realizado? (s�/no)'), nl.
pregunta(comportamiento_impulsivo_agresivo):- write('�Tiene tendencia a comportarse de forma impulsiva o agresiva? (s�/no)'), nl.
pregunta(incapacidad_relaciones_estables):- write('�Encuentra dif�cil mantener relaciones estables o duraderas? (s�/no)'), nl.

% Regla para iniciar para el diagnóstico%
diagnostico_principal :-
write('Responderé algunas preguntas para identificar posibles condiciones. Por favor, responda con "si" o "no".'), nl,
findall(Sintoma, (sintoma(Sintoma), pregunta(Sintoma), read(Respuesta), Respuesta == si), SintomasIngresados), diagnostico(SintomasIngresados).

% Regla para determinar el diagnóstico basado en los síntomas ingresados
diagnostico(Sintomas) :-
% Encontrar los trastornos con porcentaje mayor al 80%
findall(Trastorno, ( diagnostico_trastorno(Sintomas, Trastorno, Porcentaje),
     Porcentaje >= 80,
          write('El paciente podría tener: '), nl, write(Trastorno),
          write(' con un porcentaje del:'),
          write(Porcentaje), write('%'), write('.'), nl, nl),
          _),
          write('El paciente podría tener tendencias a:'), nl,
% Encontrar los trastornos en el rango del 50% al 80%
findall(Trastorno, ( diagnostico_trastorno(Sintomas, Trastorno, Porcentaje),
     Porcentaje >= 50, Porcentaje < 80,
          write(Trastorno), write('.'), nl
          ), _),
     nl, !.


%Diagnostico - TRASTORNO ANSIEDAD %
diagnostico_trastorno(Sintomas, trastorno_ansiedad, Porcentaje) :-
  SintomasAnsiedad = [preocupacion_excesiva, nerviosismo, fatiga, problemas_concentracion, irritabilidad, tension_muscular, problemas_sueno], contar_sintomas(SintomasAnsiedad, Sintomas, Conteo),
  length(SintomasAnsiedad, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - TRASTORNO DEPRESION %
diagnostico_trastorno(Sintomas, depresion, Porcentaje) :-
  SintomasDepresion = [sentimientos_tristeza, perdida_interes, cambios_apetito_peso, problemas_sueno, fatiga, sentimientos_inutilidad, pensamientos_suicidio],
  contar_sintomas(SintomasDepresion, Sintomas, Conteo),
  length(SintomasDepresion, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - TDAH %
diagnostico_trastorno(Sintomas, tdah, Porcentaje) :-
  SintomasTDAH = [dificultad_atencion, hiperactividad, impulsividad, dificultades_instrucciones],
  contar_sintomas(SintomasTDAH, Sintomas, Conteo),
  length(SintomasTDAH, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - PARKINSON %
diagnostico_trastorno(Sintomas, parkinson, Porcentaje) :-
  SintomasParkinson = [temblor_reposo, rigidez_muscular, lentitud_movimientos, problemas_equilibrio_coordinacion, dificultad_hablar_escribir],
  contar_sintomas(SintomasParkinson, Sintomas, Conteo),
  length(SintomasParkinson, Total),
  Porcentaje is (Conteo / Total) * 100.


% Diagn�stico - ALZHEIMER %
diagnostico_trastorno(Sintomas, alzheimer, Porcentaje) :-
  SintomasAlzheimer = [perdida_memoria, dificultad_palabras_conversaciones, desorientacion_espacial_temporal, cambios_estado_animo_comportamiento, dificultad_tareas_cotidianas],
  contar_sintomas(SintomasAlzheimer, Sintomas, Conteo),
  length(SintomasAlzheimer, Total),
  Porcentaje is (Conteo / Total) * 100.


% Diagn�stico - TRASTORNO BIPOLAR %
diagnostico_trastorno(Sintomas, trastorno_bipolar, Porcentaje) :-
  SintomasTBipolar = [episodios_mania, episodios_depresion, cambios_bruscos_humor_actividad],
  contar_sintomas(SintomasTBipolar, Sintomas, Conteo),
  length(SintomasTBipolar, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - TRASTORNO TOC %
diagnostico_trastorno(Sintomas, toc, Porcentaje) :-
  SintomasTOC = [obsesiones, compulsiones, reconocimiento_ineficacia_control],
  contar_sintomas(SintomasTOC, Sintomas, Conteo),
  length(SintomasTOC, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - TRASTORNO MISOFONIA %
diagnostico_trastorno(Sintomas, misofonia, Porcentaje) :-
  SintomasMisofonia = [irritabilidad_ruido, enfado, ansiedad, nauseas, sudoracion, necesidad_escapar, sonidos_desencadenantes],
  contar_sintomas(SintomasMisofonia, Sintomas, Conteo),
  length(SintomasMisofonia, Total),
  Porcentaje is (Conteo / Total) * 100.

% Diagn�stico - TRASTORNO ANTISOCIAL %
diagnostico_trastorno(Sintomas, trastorno_antisocial, Porcentaje) :-
  SintomasTAntosocial = [desprecio_normas_sociales, manipulacion_engano, falta_empatia_remordimiento, comportamiento_impulsivo_agresivo, incapacidad_relaciones_estables],
  contar_sintomas(SintomasTAntosocial, Sintomas, Conteo),
  length(SintomasTAntosocial, Total),
  Porcentaje is (Conteo / Total) * 100.


%Funcion_auxiliar_para_contar_cuantos_sintomas_de_la_lista_esten_presentes_en_los_sintomas_ingresados
contar_sintomas([], _, 0).
contar_sintomas([Sintoma|Resto], SintomasIngresados, Conteo) :-
    (member(Sintoma, SintomasIngresados) ->
        contar_sintomas(Resto, SintomasIngresados, ConteoResto), Conteo is ConteoResto + 1;
        contar_sintomas(Resto, SintomasIngresados, Conteo)).




%Regla_auxiliar_para_verificar_si_una_lista_es_un_subconjunto_de_otra_lista
subset([], _).
subset([X|Xs], Ys) :-
    member(X, Ys),
    subset(Xs, Ys).
