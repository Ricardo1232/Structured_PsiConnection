document.addEventListener("DOMContentLoaded", function() {

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


    const password = document.getElementById("contraPrac");
    const confirmPassword = document.getElementById("confirmPassword");

    // Función para verificar si las contraseñas coinciden
    function checkPasswordMatch() {
        if (password.value !== confirmPassword.value) {
            confirmPassword.setCustomValidity('Las contraseñas no coinciden');
        } else {
            confirmPassword.setCustomValidity('');
        }
    }

    // Evento para el botón de mostrar/ocultar contraseña
    document.getElementById("togglePassword").addEventListener("click", function() {
        togglePasswordVisibility('contraPrac', 'togglePassword');
    });

    document.getElementById("toggleConfirmPassword").addEventListener("click", function() {
        togglePasswordVisibility('confirmPassword', 'toggleConfirmPassword');
    });

    // Evento para verificar la coincidencia de contraseñas al escribir
    password.addEventListener('input', checkPasswordMatch);
    confirmPassword.addEventListener('input', checkPasswordMatch);

    // Función para mostrar/ocultar contraseña
    function togglePasswordVisibility(passwordId, toggleButtonId) {
        const passwordInput = document.getElementById(passwordId);
        const toggleButton = document.getElementById(toggleButtonId);
        if (passwordInput.type === "password") {
            passwordInput.type = "text";
            toggleButton.textContent = "Ocultar";
        } else {
            passwordInput.type = "password";
            toggleButton.textContent = "Mostrar";
        }
    }

    // Evento para manejar el submit del formulario
    document.getElementById('agregarPracticante').addEventListener('submit', function(event) {
        var leftInput = document.getElementById('input-editable').value;
        var rightInput = document.getElementById('input-readonly').value;
        document.getElementById('correoPrac').value = leftInput + rightInput;

        var passwordValue = document.getElementById('contraPrac').value;
        var confirmPasswordValue = document.getElementById('confirmPassword').value;

        // Patrón de la contraseña
        var passwordPattern = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+{}\[\]:;,.]).{8,}$/;

        if (!passwordPattern.test(passwordValue)) {
            alert('La contraseña no cumple con los requisitos. Debe tener al menos 8 caracteres, una mayúscula y un carácter especial');
            event.preventDefault();
        } else if (passwordValue !== confirmPasswordValue) {
            alert('Las contraseñas no coinciden.');
            event.preventDefault();
        }
    });
});
