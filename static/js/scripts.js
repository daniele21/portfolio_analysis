// static/js/scripts.js

document.addEventListener("DOMContentLoaded", function() {
    // Inizializza AOS
    AOS.init({
        duration: 800, // Durata dell'animazione in millisecondi
        easing: 'ease-in-out', // Tipo di easing
        once: true, // Animare solo una volta mentre scorri
    });

    // Funzione per aggiungere classe 'visible' quando le card entrano nel viewport (opzionale)
    function revealCards() {
        const cards = document.querySelectorAll('.card');
        const windowHeight = window.innerHeight;

        cards.forEach(card => {
            const cardTop = card.getBoundingClientRect().top;
            if (cardTop < windowHeight - 100) { // Soglia di 100px
                card.classList.add('visible');
            }
        });
    }

    // Funzione per filtrare i grafici
    function filterPlots() {
        const filter = document.getElementById('plot-filter').value;
        const rows = document.querySelectorAll('.row.mb-4');

        rows.forEach(row => {
            const title = row.querySelector('.card-header').textContent.trim().toLowerCase();
            if (filter === 'all' || title === filter.toLowerCase()) {
                row.style.display = 'block';
            } else {
                row.style.display = 'none';
            }
        });
    }

    // Esegui la funzione di reveal al caricamento della pagina
    revealCards();

    // Aggiungi un listener per lo scroll per rivelare le card
    window.addEventListener('scroll', revealCards);

    // Aggiungi listener per il filtro (se presente)
    const plotFilter = document.getElementById('plot-filter');
    if (plotFilter) {
        plotFilter.addEventListener('change', filterPlots);
    }
});
