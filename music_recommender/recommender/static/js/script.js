// script.js

let spinnerTimeout; // Variabile per memorizzare il timeout

// Funzione per mostrare lo spinner
function showSpinner() {
    document.getElementById('loading-spinner').style.display = 'flex';
}

// Funzione per nascondere lo spinner dopo un ritardo minimo
function hideSpinner() {
    clearTimeout(spinnerTimeout); // Annulla il timeout precedente
    spinnerTimeout = setTimeout(function() {
        document.getElementById('loading-spinner').style.display = 'none';
    }, 500); // Ritardo minimo di 500 ms
}

// Quando la pagina è completamente caricata, nascondi lo spinner (con ritardo)
window.addEventListener("load", function() {
    hideSpinner();
});

// Prima che la pagina venga ricaricata o cambiata, mostra lo spinner
window.addEventListener("beforeunload", function() {
    showSpinner();
});

function updateValue(slider) {
    const valueDisplay = document.getElementById(`${slider.id}-value`);
    valueDisplay.textContent = parseFloat(slider.value).toFixed(2);  // Limita il valore a due cifre decimali
}
