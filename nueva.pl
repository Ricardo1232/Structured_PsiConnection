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

% Modified diagnostic rules
diagnostico(Sintomas, Trastorno) :-
    diagnostico_trastorno(Sintomas, Trastorno).

diagnostico_trastorno(Sintomas, trastorno_ansiedad) :-
    findall(S, (member(S, [preocupacion_excesiva, nerviosismo, fatiga, problemas_concentracion, irritabilidad, tension_muscular, problemas_sueno]), member(S, Sintomas)), SintomasPresentes),
    length(SintomasPresentes, Conteo),
    length([preocupacion_excesiva, nerviosismo, fatiga, problemas_concentracion, irritabilidad, tension_muscular, problemas_sueno], Total),
    Porcentaje is (Conteo / Total) * 100,
    Porcentaje >= 80.

diagnostico_trastorno(Sintomas, depresion) :-
    SintomasDepresion = [sentimientos_tristeza, perdida_interes, cambios_apetito_peso, problemas_sueno, fatiga, sentimientos_inutilidad, pensamientos_suicidio],
    contar_sintomas(SintomasDepresion, Sintomas, Conteo),
    length(SintomasDepresion, Total),
    Porcentaje is (Conteo / Total) * 100,
    Porcentaje >= 80 .


% Add similar rules for other disorders...

% Helper predicate to check if a list is a subset of another list
subset([], _).
subset([X|Xs], Ys) :- member(X, Ys), subset(Xs, Ys).