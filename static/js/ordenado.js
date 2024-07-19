// Obtenemos la URL actual de la página
var currentUrl = window.location.pathname;
console.log(currentUrl);
// Obtenemos todos los enlaces de la barra de navegación
var navLinks = document.querySelectorAll('.nav-link.click-scroll');

// Iteramos por cada enlace y comparamos su URL con la URL actual de la página
for (var i = 0; i < navLinks.length; i++) {

    if (navLinks[i].getAttribute('href') == currentUrl) {
        // Si la URL del enlace coincide con la URL actual de la página, agregamos la clase 'active'
        navLinks[i].classList.add('active');
    }
}

// ------------------------------------ Cards index admin ------------------------------------ //
if (currentUrl == "") {
    const cards = document.querySelectorAll('.Lcard');

    for (let i = 0; i < cards.length; i++) {
        const card = cards[i];

        card.addEventListener('mouseover', () => {
            setTimeout(() => {
                card.querySelector('.Ctexto').classList.toggle('display-none');
                card.querySelector('.CtextoHover').classList.toggle('display-block');
            }, 100);
        });

        card.addEventListener('mouseout', () => {
            setTimeout(() => {
                card.querySelector('.Ctexto').classList.toggle('display-none');
                card.querySelector('.CtextoHover').classList.toggle('display-block');
            }, 100);
        });
    }
}
// ------------------------------------ End Cards index admin ------------------------------------ //









// ------------------------------------ End Agregar Supervisor ------------------------------------ //


// ------------------------------------------- Administrador ------------------------------------------- //
// ------------------------------------------- Administrador ------------------------------------------- //
// ------------------------------------------- Administrador ------------------------------------------- //

// ------------------------------------ Modificar datos administrador ------------------------------------ //
// Espera a que se haga clic en el botón "Modificar datos"
if (currentUrl == "/VerAdministrador") {
    console.log('Admin');
    $('a.btn-edit').click(function () {

        const form = document.querySelector('#editAdmin');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const idAd = form.querySelector('input[name="idAd"]');
        const Fnombre = form.querySelector('input[name="nombreAd"]');
        const FapellidoP = form.querySelector('input[name="apellidoPAd"]');
        const FapellidoM = form.querySelector('input[name="apellidoMAd"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda
        const celda2 = fila.cells[2]; // Tercera celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent;
        const valor1 = celda1.textContent;
        const valor2 = celda2.textContent;

        // ID del administrador a editar
        idAd.value = filaId

        Fnombre.value = valor0;
        FapellidoP.value = valor1;
        FapellidoM.value = valor2;
    });
    // ------------------------------------ End Modificar datos administrador ------------------------------------ //

    // ------------------------------------ Eliminar administrador ------------------------------------ //
    $('a.delete').click(function () {

        const form = document.querySelector('#deleteAdmin');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const idAd = form.querySelector('input[name="idAd"]');
        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del administrador a editar
        idAd.value = filaId;
        document.getElementById('nombreModal').textContent = valor0;
    });
    // ------------------------------------ End Eliminar administrador ------------------------------------ //

    // ------------------------------------ Correo Supervisor ------------------------------------ //
    const editableInputAdm = document.getElementById('input-editable');
    const readonlyInputAdm = document.getElementById('input-readonly');
    const combinedInputAdm = document.getElementById('correoAd');

    // Función para combinar los valores de los dos inputs
    function combineInputs() {
        const editableValue = editableInputAdm.value;
        const readonlyValue = readonlyInputAdm.value;
        combinedInputAdm.value = `${editableValue}${readonlyValue}`;
        console.log(combinedInputAdm.value);
    }

    // Asignar el valor del input combinado antes de hacer el submit
    document.getElementById('AgregarAdministrador').addEventListener('submit', function (event) {
        combineInputs();
    });

    // Escuchar cambios en los dos inputs y actualizar el input combinado
    editableInputAdm.addEventListener('input', combineInputs);
    readonlyInputAdm.addEventListener('input', combineInputs);



    // ------------------------------------ End Correo Supervisor ------------------------------------ //

    // ------------------------------------- Contraseñas administrador ------------------------------------ //
    function checkPasswordMatch() { //administrador
        var password = document.getElementById("contraAd").value;
        var confirmPassword = document.getElementById("contraAd_confirm").value;

        if (password !== confirmPassword) {
            document.getElementById("contraAd_confirm").setCustomValidity("Las contraseñas no coinciden");
        } else {
            document.getElementById("contraAd_confirm").setCustomValidity("");
        }
    }
    // ---------------------------------- End Contraseñas administrador ----------------------------------- //
}
// ------------------------------------------- End Administrador ------------------------------------------- //
// ------------------------------------------- End Administrador ------------------------------------------- //
// ------------------------------------------- End Administrador ------------------------------------------- //



// ------------------------------------------- Supervisor ------------------------------------------- //
// ------------------------------------------- Supervisor ------------------------------------------- //
// ------------------------------------------- Supervisor ------------------------------------------- //
if (currentUrl == "/VerSupervisor") {
    console.log('Sup');

    // ------------------------------------ Modificar datos Supervisor ------------------------------------ //
    $('a.btn-edit').click(function () {

        const form = document.querySelector('#editSup');
        const filaId = $(this).closest('tr').data('id');

        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idSup"]');
        const FnombreSup = form.querySelector('input[name="nombreSup"]');
        const FapellidoPSup = form.querySelector('input[name="apellidoPSup"]');
        const FapellidoMSup = form.querySelector('input[name="apellidoMSup"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda
        const celda2 = fila.cells[2]; // Tercera celda


        // Obtener los valores de las celdas
        const valor0 = celda0.textContent;
        const valor1 = celda1.textContent;
        const valor2 = celda2.textContent;

        // ID del administrador a editar
        id.value = filaId;

        FnombreSup.value = valor0;
        FapellidoPSup.value = valor1;
        FapellidoMSup.value = valor2;
    });
    // ------------------------------------ End Modificar datos Supervisor ------------------------------------ //

    // ------------------------------------ Eliminar Supervisor ------------------------------------ //
    $('a.delete').click(function () {

        const form = document.querySelector('#deleteSup');
        const table = document.querySelector('#SupTable');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idSup"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del supervisor a editar
        id.value = filaId;
        document.getElementById('nombreModal').textContent = valor0;

    });
    // ------------------------------------ End Eliminar Supervisor ------------------------------------ //

    // ------------------------------------ Correo Supervisor ------------------------------------ //
    const editableInputSup = document.getElementById('input-editable');
    const readonlyInputSup = document.getElementById('input-readonly');
    const combinedInputSup = document.getElementById('correoSup');

    // Función para combinar los valores de los dos inputs
    function combineInputs() {
        const editableValue = editableInputSup.value;
        const readonlyValue = readonlyInputSup.value;
        combinedInputSup.value = `${editableValue}${readonlyValue}`;
        console.log(combinedInputSup.value);
    }

    // Asignar el valor del input combinado antes de hacer el submit
    document.getElementById('AgregarSupervisor').addEventListener('submit', function (event) {
        combineInputs();
    });

    // Escuchar cambios en los dos inputs y actualizar el input combinado
    editableInputSup.addEventListener('input', combineInputs);
    readonlyInputSup.addEventListener('input', combineInputs);


    // ------------------------------------ End Correo Supervisor ------------------------------------ //


    // ----------------------------------------- Supervisor Contraseña ----------------------------------------- //
    function checkPasswordMatch() { //supervisor
        var password = document.getElementById("contraSup").value;
        var confirmPassword = document.getElementById("contraSup_confirm").value;

        if (password !== confirmPassword) {
            document.getElementById("contraSup_confirm").setCustomValidity("Las contraseñas no coinciden");
        } else {
            document.getElementById("contraSup_confirm").setCustomValidity("");
        }
    }
    // ----------------------------------------- End Supervisor Contraseña ----------------------------------------- //
}
// ------------------------------------------- End Supervisor ------------------------------------------- //
// ------------------------------------------- End Supervisor ------------------------------------------- //
// ------------------------------------------- End Supervisor ------------------------------------------- //



// ------------------------------------------- Practicante ------------------------------------------- //
// ------------------------------------------- Practicante ------------------------------------------- //
// ------------------------------------------- Practicante ------------------------------------------- //

// Espera a que se haga clic en el botón "Modificar datos"

if ((currentUrl == "/VerPracticantesAdm") || (currentUrl == "/IndexSupervisor")) {
    console.log('Prac');
    // ------------------------------------ Modificar datos practicante ------------------------------------ //
    $('a.btn-edit').click(function () {

        const form = document.querySelector('#editPrac');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idPrac"]');
        const FnombrePrac = form.querySelector('input[name="nombrePrac"]');
        const FapellidoPPrac = form.querySelector('input[name="apellidoPPrac"]');
        const FapellidoMPrac = form.querySelector('input[name="apellidoMPrac"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda
        const celda2 = fila.cells[2]; // Tercera celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent;
        const valor1 = celda1.textContent;
        const valor2 = celda2.textContent;

        // ID del administrador a editar
        id.value = filaId;

        FnombrePrac.value = valor0;
        FapellidoPPrac.value = valor1;
        FapellidoMPrac.value = valor2;
    });
    // ------------------------------------ End Modificar datos practicante  ------------------------------------ //

    // ------------------------------------ Eliminar practicante  ------------------------------------ //
    $('a.delete').click(function () {

        const form = document.querySelector('#deletePrac');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idPrac"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del supervisor a editar
        id.value = filaId;
        document.getElementById('nombreModal').textContent = valor0;
    });
}
// ------------------------------------ End Eliminar practicante  ------------------------------------ //

// ------------------------------------------- End practicante  ------------------------------------------- //
// ------------------------------------------- End practicante  ------------------------------------------- //
// ------------------------------------------- End practicante  ------------------------------------------- //




// ------------------------------------------- Paciente ------------------------------------------- //
// ------------------------------------------- Paciente ------------------------------------------- //
// ------------------------------------------- Paciente ------------------------------------------- //

// Espera a que se haga clic en el botón "Modificar datos"

if (currentUrl == "/VerPacientesAdm") {
    console.log('Paci');
    // ------------------------------------ Modificar datos Paciente ------------------------------------ //
    $('a.btn-edit').click(function () {

        const form = document.querySelector('#editPaci');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idPaci"]');
        const FnombrePac = form.querySelector('input[name="nombrePaci"]');
        const FapellidoPPac = form.querySelector('input[name="apellidoPPaci"]');
        const FapellidoMPac = form.querySelector('input[name="apellidoMPaci"]');

        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda
        const celda2 = fila.cells[2]; // Tercera celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent;
        const valor1 = celda1.textContent;
        const valor2 = celda2.textContent;

        // ID del Paciente a editar
        id.value = filaId;

        FnombrePac.value = valor0;
        FapellidoPPac.value = valor1;
        FapellidoMPac.value = valor2;
    });
    // ------------------------------------ End Modificar datos Paciente  ------------------------------------ //

    // ------------------------------------ Eliminar Paciente  ------------------------------------ //
    $('a.delete').click(function () {

        const form = document.querySelector('#deletePaci');
        const filaId = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${filaId}"]`);

        const id = form.querySelector('input[name="idPaci"]');
        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda

        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del Paciente a editar
        id.value = filaId;
        document.getElementById('nombreModal').textContent = valor0;
    });
}
// ------------------------------------ End Eliminar Paciente  ------------------------------------ //

// ------------------------------------------- End Paciente  ------------------------------------------- //
// ------------------------------------------- End Paciente  ------------------------------------------- //
// ------------------------------------------- End Paciente  ------------------------------------------- //

if (currentUrl == "/AgregarPracticante") {
    console.log('Agregar practicante');
    // ------------------------------------ Correo practicante ------------------------------------ //
    const editableInputPrac = document.getElementById('input-editable');
    const readonlyInputPrac = document.getElementById('input-readonly');
    const combinedInputPrac = document.getElementById('correoPrac');

    // Función para combinar los valores de los dos inputs
    function combineInputs() {
        const editableValue = editableInputPrac.value;
        const readonlyValue = readonlyInputPrac.value;
        combinedInputPrac.value = `${editableValue}${readonlyValue}`;
        console.log(combinedInputPrac.value);
    }

    // Asignar el valor del input combinado antes de hacer el submit
    document.getElementById('agregarPacticante').addEventListener('submit', function (event) {
        combineInputs();
    });

    // Escuchar cambios en los dos inputs y actualizar el input combinado
    editableInputPrac.addEventListener('input', combineInputs);
    readonlyInputPrac.addEventListener('input', combineInputs);


    // ------------------------------------ End Correo Practicante ------------------------------------ //

    // ------------------------------------ Contraseña practicante ------------------------------------ //

    function checkPasswordMatch() {
        var password = document.getElementById("contraPrac").value;
        var confirmPassword = document.getElementById("confirmPassword").value;

        if (password !== confirmPassword) {
            document.getElementById("confirmPassword").setCustomValidity("Las contraseñas no coinciden");
        } else {
            document.getElementById("confirmPassword").setCustomValidity("");
        }
    }
    // ------------------------------------ End Contraseña practicante ------------------------------------ //

    // ------------------------------------ Fecha practicante ------------------------------------ //
  console.log(currentUrl);
  // Obtener la fecha actual
  var fechaActual = new Date();

  // Restar 18 años a la fecha actual
  var fechaHace18Anios = new Date(fechaActual.getFullYear() - 18, fechaActual.getMonth(), fechaActual.getDate());

  // Obtener el campo de entrada de fecha
  var fechaInput = document.getElementById('fechaNacPrac');

  // Establecer la fecha máxima permitida en el campo de entrada
  fechaInput.max = fechaHace18Anios.toISOString().split('T')[0];

  // Validar la fecha seleccionada cada vez que cambie el valor del campo de entrada
  fechaInput.addEventListener('input', function () {
    var fechaSeleccionada = new Date(this.value);

    if (fechaSeleccionada > fechaHace18Anios) {
      // La fecha seleccionada es menor a 18 años antes de la fecha actual
      console.log('La fecha seleccionada no cumple con la restricción de edad mínima.');
      // Aquí puedes agregar acciones adicionales, como mostrar un mensaje de error al usuario.
    } else {
      // La fecha seleccionada es mayor o igual a 18 años antes de la fecha actual
      console.log('La fecha seleccionada cumple con la restricción de edad mínima.');
      // Aquí puedes realizar las acciones adicionales necesarias en caso de que la fecha sea válida.
    }
  });
}


// ----------------------------------------- Ocultar ----------------------------------------- //
/* para boton de ocultar contraseña */  //administrador
function togglePasswordVisibility(inputId, buttonId) {
    var passwordInput = document.getElementById(inputId);
    var toggleButton = document.getElementById(buttonId);

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        toggleButton.textContent = "Ocultar";
    } else {
        passwordInput.type = "password";
        toggleButton.textContent = "Mostrar";
    }
}
// ----------------------------------------- End Ocultar ----------------------------------------- //

// ------------------------------------ Modal eliminar cita paciente ------------------------------------ //
if (currentUrl == "/IndexPacientes") {
    $('a.delete').click(function () {
        console.log("Hola");
        const form = document.querySelector('#borrarCita');
        const cita = $(this).closest('tr').data('id');
        const eventoId = $(this).closest('tr').data('evento-id');
        const fechaCita = $(this).closest('tr').data('fecha-cita');
        const horaCita = $(this).closest('tr').data('hora-cita');
        const fila = document.querySelector(`tr[data-id="${cita}"]`);

        const FidCita = form.querySelector('input[name="idCita"]');
        const FideventoCita = form.querySelector('input[name="eventoCita"]');
        const FidfechaCita = form.querySelector('input[name="fechaCita"]');
        const FidhoraCita = form.querySelector('input[name="horaCita"]');
        


        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda


        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del administrador a editar
        FidCita.value = cita;
        FideventoCita.value = eventoId
        FidfechaCita.value = fechaCita
        FidhoraCita.value = horaCita
        
        document.getElementById('nombreModal').textContent = valor0;
        

    });
}
// ------------------------------------ End Modal eliminar cita paciente------------------------------------ //



if (currentUrl == "/AgendarCita") {
    console.log('Cita');
    // ----------------------------  Fecha ---------------------------- //
    // Obtenemos la fecha actual
    var today = new Date().toISOString().split('T')[0];

    // Obtenemos el elemento del input de fecha
    var dateInput = document.getElementById('fechaCita');

    // Establecemos la fecha mínima permitida como la fecha actual
    dateInput.setAttribute('min', today);

    // ---------------------------- End fecha ---------------------------- //

    // ---------------------------- Hora ---------------------------- //
    const timeInput = document.getElementById('horaCita');

    timeInput.addEventListener('input', () => {
        const time = timeInput.value;
        const parts = time.split(':');
        timeInput.value = `${parts[0]}:00`;
    });

    const timeError = document.getElementById('time-error');


    timeInput.addEventListener('change', () => {
        const selectedTime = timeInput.value;
        const selectedHour = parseInt(selectedTime.split(':')[0], 10);

        if (selectedHour < 8 || selectedHour > 20) {
            timeInput.setCustomValidity('Elija una hora valida');
            timeInput.value = '';
            timeError.style.display = 'block';

            const optionElements = timeInput.querySelectorAll('option');
            optionElements.forEach(option => {
                const hour = parseInt(option.value.split(':')[0], 10);
                if (hour < 8 || hour > 20) {
                    option.disabled = true;
                } else {
                    option.disabled = false;
                }
            });
        } else {
            timeInput.setCustomValidity('');
            timeError.style.display = 'none';

            const optionElements = timeInput.querySelectorAll('option');
            optionElements.forEach(option => {
                option.disabled = false;
            });
        }
    });

    const form = document.getElementById('elegir');
    const modalidadSelect = document.getElementById('tipoCita');

    form.addEventListener("submit", (event) => {
        if (modalidadSelect.value === "") {
            event.preventDefault();
        }
    });
    //---------------------------- End hora ---------------------------- //

    // --------------------------- Modal agenda cita --------------------------- //
    $('a.link-card-practicante').click(function () {

        const form = document.querySelector('#elegir');
        // Obtener el valor del atributo data-id
        const tupla = $(this).data('id');

        // Dividir el valor por la coma
        var values = tupla.split(",");
        // Acceder a los valores que necesitas
        var idPrac = values[0];
        var correoPrac = values[1];

        // Inputs
        const FidPrac = form.querySelector('input[name="idPrac"]');
        const FnombrePrac = form.querySelector('input[name="nombrePrac"]');
        const FapellidoP = form.querySelector('input[name="apellidoPPrac"]');
        const FapellidoM = form.querySelector('input[name="apellidoMPrac"]');

        // Inputs correo
        const FcorreoPrac = form.querySelector('input[name="correoPrac"]');

        // Asignacion id
        FidPrac.value = idPrac;

        // Asignacion correo
        FcorreoPrac.value = correoPrac;

        // Texto correo
        const correoPaci = document.getElementById('correoPaci').textContent;

        // Inputs paciente
        const FidPaci = form.querySelector('input[name="idPaci"]');
        const FcorreoPaci = form.querySelector('input[name="correoPaci"]');

        // Asignacion paciente
        FidPaci.value = $(this).find('#idPaci').text();
        FcorreoPaci.value = correoPaci;

        // Texto nombre y apellidos
        const nombrePrac = $(this).find('#nombrePrac').text();
        const apellidoPPrac = $(this).find('#apellidoPPrac').text();
        const apellidoMPrac = $(this).find('#apellidoMPrac').text();

        // Asignacion nombre y apellidos
        document.getElementById('Fnombre').textContent = nombrePrac;
        document.getElementById('FapellidoPPrac').textContent = apellidoPPrac;

        FnombrePrac.value = nombrePrac;
        FapellidoP.value = apellidoPPrac;
        FapellidoM.value = apellidoMPrac;

    });

}
// --------------------------- End Modal agenda cita --------------------------- //

// ------------------------------------ Modal eliminar cita practicante ------------------------------------ //
if (currentUrl == "/IndexPracticantes") {
    $('a.delete').click(function () {
        console.log("Estoy en borrar cita practicante")
        const form = document.querySelector('#borrarCita');
        const cita = $(this).closest('tr').data('id');
        const fila = document.querySelector(`tr[data-id="${cita}"]`);
        const fechaCita = $(this).closest('tr').data('fecha-cita');
        const horaCita = $(this).closest('tr').data('hora-cita');


        const FidCita = form.querySelector('input[name="idCita"]');
        const FidfechaCita = form.querySelector('input[name="fechaCita"]');
        const FidhoraCita = form.querySelector('input[name="horaCita"]');


        // Acceder a las celdas específicas por índice
        const celda0 = fila.cells[0]; // Primera celda
        const celda1 = fila.cells[1]; // Segunda celda


        // Obtener los valores de las celdas
        const valor0 = celda0.textContent + ' ' + celda1.textContent;
        // ID del administrador a editar
        FidCita.value = cita;
        FidfechaCita.value = fechaCita;
        FidhoraCita.value = horaCita;

        document.getElementById('nombreModal').textContent = valor0;

    });
}
// ------------------------------------ End Modal eliminar cita practicante ------------------------------------ //

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