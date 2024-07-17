// Obtenemos la URL actual de la página
var currentUrl = window.location.pathname;
console.log(currentUrl );
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
if (currentUrl == "/templates/admin/index_admin.html") {
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

// ------------------------------------ Agregar practicante ------------------------------------ //
if (currentUrl == "/templates/supervisor/agregar_practicante.html") {
  console.log(currentUrl);
  const editableInput = document.getElementById('input-editable');
  const readonlyInput = document.getElementById('input-readonly');
  const combinedInput = document.getElementById('correoPrac');

  // Función para combinar los valores de los dos inputs
  function combineInputs() {
    const editableValue = editableInput.value;
    const readonlyValue = readonlyInput.value;
    combinedInput.value = `${editableValue}${readonlyValue}`;
    console.log(combinedInput.value);
  }

  // Asignar el valor del input combinado antes de hacer el submit
  document.getElementById('agregarPacticante').addEventListener('submit', function (event) {
    combineInputs();
  });

  // Escuchar cambios en los dos inputs y actualizar el input combinado
  editableInput.addEventListener('input', combineInputs);
  readonlyInput.addEventListener('input', combineInputs);

  const formulario = document.getElementById('agregarPacticante');

  formulario.addEventListener('submit', function (event) {
    event.preventDefault(); // Evita que se realice el submit por defecto del formulario
  });
}

// ------------------------------------ End Agregar practicante ------------------------------------ //



// ------------------------------------  loadMoreBtn ------------------------------------ //
// if (currentUrl == "/templates/paciente/agenda_cita.html") {
//   // Seleccionamos el botón y obtenemos las clases originales
//   const loadMoreBtn = document.querySelector('#loadMoreBtn');

//   // Función para intercambiar las clases del botón
//   function toggleClasses() {

//     if (window.innerWidth <= 993) {
//       if (window.innerWidth >= 723) {
//         loadMoreBtn.classList.remove('w-25');
//         loadMoreBtn.classList.add('w-sm-25');
//         loadMoreBtn.classList.remove('w-sm-100');
//       } else if (window.innerWidth < 723) {
//         loadMoreBtn.classList.add('w-sm-100');
//       }
//     } else if (window.innerWidth > 993) {
//       loadMoreBtn.classList.remove('w-sm-25');
//       loadMoreBtn.classList.add('w-25');
//     }
//   }
//   // Al cargar la página y al cambiar el tamaño de la pantalla, llamamos a la función toggleClasses
//   window.addEventListener('load', toggleClasses);
//   window.addEventListener('resize', toggleClasses);
// }
// ------------------------------------ End loadMoreBtn ------------------------------------ //


// --------------------------- Fecha hora --------------------------- //

if (currentUrl == "/AgendarCita") {

  // ---------------------------- Validacion fecha ---------------------------- //
  // Obtenemos la fecha actual
  var today = new Date().toISOString().split('T')[0];

  // Obtenemos el elemento del input de fecha
  var dateInput = document.getElementById('fechaCita');

  // Establecemos la fecha mínima permitida como la fecha actual
  dateInput.setAttribute('min', today);

  // ---------------------------- End Validacion fecha ---------------------------- //

  // ---------------------------- Validacion hora ---------------------------- //
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
  //---------------------------- End Validacion hora ---------------------------- //
  // --------------------------- End Fecha hora --------------------------- //

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
