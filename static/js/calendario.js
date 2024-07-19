document.addEventListener('DOMContentLoaded', function () {
    const calendar = document.getElementById('calendar').getElementsByTagName('tbody')[0];
    const personSelect = document.getElementById('person-select');
    const prevWeekButton = document.getElementById('prev-week');
    const nextWeekButton = document.getElementById('next-week');
    const monthYearElement = document.getElementById('month-year');
    
    let currentDate = new Date();
    const maxFutureDate = new Date();
    maxFutureDate.setMonth(maxFutureDate.getMonth() + 2);

    const events = [
        {
            "name": "Persona 1",
            "schedule": [
                { "date": "2024-07-22", "available": ["08:00", "09:00", "10:00"], "notAvailable": ["11:00", "12:00"] },
                { "date": "2024-07-23", "available": ["08:00", "09:00", "10:00"], "notAvailable": ["11:00", "12:00"] },
                { "date": "2024-07-24", "available": ["08:00", "09:00", "10:00"], "notAvailable": ["11:00", "12:00"] },
                { "date": "2024-07-25", "available": ["08:00", "09:00", "10:00"], "notAvailable": ["11:00", "12:00"] },
                { "date": "2024-07-26", "available": ["08:00", "09:00", "10:00"], "notAvailable": ["11:00", "12:00"] }
            ]
        },
        {
            "name": "Persona 2",
            "schedule": [
                { "date": "2024-07-22", "available": ["09:00", "10:00"], "notAvailable": ["08:00", "11:00"] },
                { "date": "2024-07-23", "available": ["09:00", "10:00"], "notAvailable": ["08:00", "11:00"] },
                { "date": "2024-07-24", "available": ["09:00", "10:00"], "notAvailable": ["08:00", "11:00"] },
                { "date": "2024-07-25", "available": ["09:00", "10:00"], "notAvailable": ["08:00", "11:00"] },
                { "date": "2024-07-26", "available": ["09:00", "10:00"], "notAvailable": ["08:00", "11:00"] }
            ]
        }
    ];

    function updatePersonSelect() {
        personSelect.innerHTML = '';
        events.forEach(event => {
            const option = document.createElement('option');
            option.value = event.name;
            option.textContent = event.name;
            personSelect.appendChild(option);
        });
        if (events.length > 0) {
            personSelect.value = events[0].name; // Seleccionar la primera persona por defecto
            generateCalendar();
        }
    }

    function generateCalendar() {
        const selectedPerson = personSelect.value;
        const personEvents = events.find(event => event.name === selectedPerson);
        if (!personEvents) {
            console.error('No se encontró la persona seleccionada.');
            return;
        }

        const schedule = personEvents.schedule || [];
        console.log('Schedule:', schedule);

        calendar.innerHTML = '';
        const startOfWeek = getStartOfWeek(currentDate);
        const hours = Array.from({ length: 12 }, (_, i) => `${i + 8}:00`);
        const daysOfWeek = [];

        for (let i = 0; i < 5; i++) {
            const currentDay = new Date(startOfWeek);
            currentDay.setDate(currentDay.getDate() + i);
            const dayString = currentDay.toISOString().split('T')[0];
            const displayDayString = currentDay.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' });
            daysOfWeek.push({ date: dayString, day: displayDayString });
        }

        // Crear encabezado de días
        const headerRow = document.createElement('tr');
        const hourHeader = document.createElement('th');
        hourHeader.textContent = 'Hora';
        headerRow.appendChild(hourHeader);

        daysOfWeek.forEach(day => {
            const dayCell = document.createElement('th');
            dayCell.classList.add('day-header');
            dayCell.innerHTML = `${day.day}`;
            headerRow.appendChild(dayCell);
        });

        calendar.appendChild(headerRow);

        // Crear filas de horas
        hours.forEach(hour => {
            const row = document.createElement('tr');
            const timeCell = document.createElement('td');
            timeCell.textContent = hour;
            row.appendChild(timeCell);

            daysOfWeek.forEach(day => {
                const cell = document.createElement('td');
                cell.classList.add('hour-cell');
                const dateString = day.date;

                const daySchedule = schedule.find(sch => sch.date === dateString);
                console.log('Searching for:', dateString);
                console.log('Day Schedule:', daySchedule);

                if (daySchedule) {
                    if (daySchedule.available.includes(hour)) {
                        cell.classList.add('available');
                    } else if (daySchedule.notAvailable.includes(hour)) {
                        cell.classList.add('not-available');
                    }
                } else {
                    cell.classList.add('not-available'); // No disponible por defecto
                }

                row.appendChild(cell);
            });

            calendar.appendChild(row);
        });

        // Actualizar encabezado de mes y año
        monthYearElement.textContent = `${currentDate.toLocaleDateString('es-ES', { month: 'long', year: 'numeric' })}`;
    }

    function getStartOfWeek(date) {
        const start = new Date(date);
        start.setDate(start.getDate() - start.getDay() + 1); // Lunes
        return start;
    }

    function changeWeek(offset) {
        const newDate = new Date(currentDate);
        newDate.setDate(currentDate.getDate() + (7 * offset));
        if (newDate >= new Date() && newDate <= maxFutureDate) {
            currentDate = newDate;
            generateCalendar();
        }
    }

    personSelect.addEventListener('change', generateCalendar);
    prevWeekButton.addEventListener('click', () => changeWeek(-1));
    nextWeekButton.addEventListener('click', () => changeWeek(1));

    // Inicializar opciones de personas y calendario
    updatePersonSelect();
});
