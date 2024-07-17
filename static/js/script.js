/*global $, document, window, setTimeout, navigator, console, location*/
$(document).ready(function () {

    'use strict';

    var usernameError = true,
        namepError = true,
        namemError = true,
        sexoError = true,
        emailError = true,
        passwordError = true,
        passConfirm = true;

    // Detect browser for css purpose
    if (navigator.userAgent.toLowerCase().indexOf('firefox') > -1) {
        $('.form form label').addClass('fontSwitch');
    }

    // Label effect
    $('input').focus(function () {

        $(this).siblings('label').addClass('active');
    });

    // Form validation
    $('input').blur(function () {

        // User Name
        if ($(this).hasClass('name')) {
            if ($(this).val().length === 0) {
                $(this).siblings('span.error').text('Porfavor ingrese su nombre completo').fadeIn().parent('.form-group').addClass('hasError');
                usernameError = true;
            } else if ($(this).val().length > 1 && $(this).val().length <= 6) {
                $(this).siblings('span.error').text('Porfavor ingrese al menos 6 caracteres').fadeIn().parent('.form-group').addClass('hasError');
                usernameError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                usernameError = false;
            }
        }

        // Apellido Paterno
        if ($(this).hasClass('apellidop')) {
            if ($(this).val().length === 0) {
                $(this).siblings('span.error').text('Porfavor ingrese su Apellido Paterno').fadeIn().parent('.form-group').addClass('hasError');
                namepError = true;
            } else if ($(this).val().length > 1 && $(this).val().length <= 1) {
                $(this).siblings('span.error').text('Porfavor ingrese al menos 6 caracteres').fadeIn().parent('.form-group').addClass('hasError');
                namepError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                namepError = false;
            }
        }

         // Apellido Materno
        if ($(this).hasClass('apellidom')) {
            if ($(this).val().length === 0) {
                $(this).siblings('span.error').text('Porfavor ingrese su Apellido Materno').fadeIn().parent('.form-group').addClass('hasError');
                namemError = true;
            } else if ($(this).val().length > 1 && $(this).val().length <= 1) {
                $(this).siblings('span.error').text('Porfavor ingrese al menos 6 caracteres').fadeIn().parent('.form-group').addClass('hasError');
                namemError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                namemError = false;
            }
        }


        //Fecha_de_nacimiento, el tipo de input es un date
        if ($(this).hasClass('fecha_nacimiento')) {
            if ($(this).val().length === 0) {
                $(this).siblings('span.error').text('Porfavor ingrese su fecha de nacimiento').fadeIn().parent('.form-group').addClass('hasError');
                fechaError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                fechaError = false;
            }
        }


        // Sexo, el valor de sexo es un select
        if ($(this).hasClass('sexo')) {
            if ($(this).val().length === 0) {
                $(this).siblings('span.error').text('Porfavor seleccione su sexo').fadeIn().parent('.form-group').addClass('hasError');
                sexoError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                sexoError = false;
            }
        }

        // Email
        if ($(this).hasClass('email')) {
            if ($(this).val().length == '') {
                $(this).siblings('span.error').text('Porfavor ingrese su correo electronico').fadeIn().parent('.form-group').addClass('hasError');
                emailError = true;
            } else if (!/^[^\s@]+@(udg\.com\.mx|alumnos\.udg\.mx|academicos\.udg\.mx)$/i.test($(this).val())) {
                $(this).siblings('span.error').text('Porfavor ingrese un correo electrónico válido con uno de los dominios permitidos').fadeIn().parent('.form-group').addClass('hasError');
                emailError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                emailError = false;
            }
        }
        // PassWord
        if ($(this).hasClass('pass')) {
            if ($(this).val().length < 8) {
                $(this).siblings('span.error').text('Porfavor ingrese al menos 8 caracteres').fadeIn().parent('.form-group').addClass('hasError');
                passwordError = true;
            } else {
                $(this).siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
                passwordError = false;
            }
        }

        // PassWord confirmation
        if ($('.pass').val() !== $('.passConfirm').val()) {
            $('.passConfirm').siblings('.error').text('Las contraseñas no coinciden').fadeIn().parent('.form-group').addClass('hasError');
            passConfirm = false;
        } else {
            $('.passConfirm').siblings('.error').text('').fadeOut().parent('.form-group').removeClass('hasError');
            passConfirm = false;
        }

        // label effect
        if ($(this).val().length > 0) {
            $(this).siblings('label').addClass('active');
        } else {
            $(this).siblings('label').removeClass('active');
        }
    });


    // form switch
    $('a.switch').click(function (e) {
        $(this).toggleClass('active');
        e.preventDefault();

        if ($('a.switch').hasClass('active')) {
            $(this).parents('.form-peice').addClass('switched').siblings('.form-peice').removeClass('switched');
        } else {
            $(this).parents('.form-peice').removeClass('switched').siblings('.form-peice').addClass('switched');
        }
    });


    //Alertas

    $(document).ready(function () {
        console.log('ready');
        // Mostrar las alertas con animación
        $('.flashes li').each(function () {
            $(this).addClass('show');
        });

        // Ocultar las alertas con animación después de 5 segundos
        setTimeout(function () {
            $('.flashes li').removeClass('show');
            setTimeout(function () {
                $('.flashes li').remove();
            }, 500);
        }, 5000);
    });

    // Crear una variable que almacene el elemento input
    var inputFecha = document.getElementById("fecha_nacimiento");

    // Añadir un evento "change" al elemento input
    inputFecha.addEventListener("change", function() {
    // Dentro de la función, comprobar si el valor es una cadena vacía
    if (inputFecha.value === "") {
        // Si es así, cambiar el estilo del elemento input a transparente
        inputFecha.style.color = "transparent";
    } else {
        // Si no, cambiar el estilo del elemento input a rojo
        inputFecha.style.color = "black";
    }
    });




});




/* Form submit
$('form.signup-form').submit(function (event) {
    

    if (usernameError == true || emailError == true || passwordError == true || passConfirm == true) {
        $('.name, .email, .pass, .passConfirm').blur();
    } else {
        // Obtener los datos del formulario
        var formData = {
            'email': $('input[name=email]').val(),
            'password': $('input[name=password]').val(),
            'action': 'register'
        };
        
        // Enviar los datos del formulario a la ruta /pythonlogin
        $.ajax({
            type: 'POST',
            url: '/pythonlogin',
            data: formData,
            dataType: 'json',
            encode: true
        }).done(function(data) {
            // Manejar la respuesta del servidor
            // ...
        });
        
            $('.signup, .login').addClass('switched');

            setTimeout(function () { $('.signup, .login').hide(); }, 700);
            setTimeout(function () { $('.brand').addClass('active'); }, 300);
            setTimeout(function () { $('.heading').addClass('active'); }, 600);
            setTimeout(function () { $('.success-msg p').addClass('active'); }, 900);
            setTimeout(function () { $('.success-msg a').addClass('active'); }, 1050);
            setTimeout(function () { $('.form').hide(); }, 700);
        });
    }
});
Mensaje recibido.
*/