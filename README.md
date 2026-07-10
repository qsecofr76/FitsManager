# FitsManager

FitsManager è un'applicazione desktop leggera per Windows dedicata alla visualizzazione, all'analisi rapida, alla calibrazione e alla misurazione fotometrica di file astronomici in formato FITS (Flexible Image Transport System). Integra funzioni avanzate di allineamento e coordinate celesti (WCS), interrogazioni di cataloghi stellari, calibrazione fotometrica ed estrazione di dati fisici in tempo reale.

*(English version below)*

---

## Caratteristiche Principali (Italiano)

### 1. Correzione dei Livelli, Bilanciamento del Bianco e Filtri
* **Color Balance & Levels (Post-Stretched)**: Regolazione fine in tempo reale di *Red Shift*, *Green Shift*, *Blue Shift*, *Luminosità* e *Contrasto* per calibrare i colori e far risaltare i dettagli più deboli.
* **Filtro di Pulizia (Denoise/Smooth)**: Slider dedicato per applicare un filtro kernel gaussiano dinamico per ammorbidire il rumore di fondo dell'immagine.
* **Autostretch ed Curve di Livello**: Stretching logaritmico automatico (basato sulla deviazione standard mediana) per immagini grezze non stirate.
* **Campionamento del Colore**: Calibrazione interattiva sul nero (fondo cielo) e sul bianco (stelle calde) selezionando direttamente i pixel di riferimento a schermo.

### 2. Marcatura Oggetti Celesti ed Annotazioni
* **Annotazioni Personalizzate**: Clicca su qualsiasi oggetto per posizionare un mirino di precisione (aperto al centro) inserendo note di testo personalizzate.
* **Integrazione coordinate J2000**: Il programma calcola e allega automaticamente le coordinate celesti RA/Dec J2000 al testo dell'annotazione.
* **Tracciamento Asteroidi**: Interroga in automatico il database IMCCE SkyBoT per visualizzare con **rombi arancioni** la posizione degli asteroidi con $Mag < 21$ presenti nell'inquadratura al momento dello scatto.

### 3. Fotometria e Calcolo della Magnitudine (Flusso di Lavoro Separato)
* **Separazione tra Calibrazione e Misura**:
  * **Mark Calib Stars**: Permette di marcare e aggiungere le stelle di calibrazione stabili (cerchi azzurri) escludendo le stelle variabili. Cliccando una stella già inserita, questa viene deselezionata/rimossa (comportamento Toggle).
  * **Measure Target**: Permette di calcolare la magnitudine di qualsiasi stella (incluse le variabili e quelle del catalogo).
* **Calibrazione Automatica**: Suggerimento immediato tramite MessageBox per avviare l'**Auto-Calibrazione fotometrica** subito dopo il download del catalogo.
* **Interpolazione del Colore (FIT a Colori)**: Nei FIT RGB, il programma misura il flusso nei canali Rosso e Blu per ricavare l'indice di colore strumentale:
  $$CI_{\text{str}} = -2.5 \log_{10}(F_B / F_R)$$
  Utilizza le stelle di calibrazione per interpolare la retta strumentale ed estrarre silenziosamente e con massima precisione la magnitudine fisica reale di transitori (senza aprire popup per stelle note del catalogo).
* **Classificazione Probabilistica del Target**: Mostra nel pop-up dei risultati la stima percentuale del tipo di stella misurata in base al colore calcolato (*Nova, Supernova tipo Ia, Asteroide/Nana Gialla, Nana Rossa, Stella Gigante Blu*).

### 4. Confronto con Immagini DSS (con Fade)
* **DSS Reference Sky Overlay**: Scarica in background e memorizza in cache le lastre reali DSS2 (red plate) da STScI.
* **Fade Slider**: Permette di sfumare in tempo reale (Cross-Fade) tra la tua foto calde e l'immagine reale del cielo DSS perfettamente allineata tramite WCS.

### 5. Plate Solving e Riscrittura Coordinate WCS
* **Plate Solving Multiplo**: Supporta la risoluzione tramite astrometri locali/Vizier (astroalign) o tramite il risolutore online di **Nova.astrometry.net**.
* **Salvataggio WCS**: Riscrive le coordinate risolte direttamente nell'intestazione (Header) del file FITS originale senza perdita di dati.

### 6. Ricerca Stella su GAIA (Gaia Search Mode)
* **Gaia Star Search Mode**: Cliccando su una stella in questa modalità, interroga il catalogo GAIA DR3 per ricavare magnitudine, moto proprio, parallasse e indice di colore.
* **Riferimenti Grafici ed Aladin Lite**: Evidenzia il bersaglio cliccato con un **quadrato verde** e la coordinata esatta restituita da Gaia con una **X rossa**, fornendo un link interattivo cliccabile per aprire l'area su **Aladin Lite** (mappa celeste interattiva nel browser).

---

## Installazione ed Avvio da Sorgente
1. Installa Python 3.8+ sul tuo computer.
2. Clona la repository ed entra nella cartella:
   ```bash
   git clone https://github.com/qsecofr76/FitsManager.git
   cd FitsManager
   ```
3. Installa le dipendenze:
   ```bash
   python install_requirements.py
   ```
4. Avvia l'applicazione:
   ```bash
   python fits_manager.py
   ```

---

## Rilascio Windows (EXE Pronto all'Uso)
Per gli utenti Windows che preferiscono non installare Python o configurare l'ambiente di sviluppo, è disponibile un pacchetto standalone già pronto.
Potete scaricare l'archivio ZIP precompilato direttamente dalla sezione **Releases** di questo repository GitHub.

Una volta scaricato lo ZIP:
1. Estraete il contenuto in una cartella a vostra scelta.
2. Eseguite il file **`FitsManager.exe`** all'interno della cartella estratta.

---
---

## Main Features (English)

### 1. Levels Adjustment, Color Balance & Filters
* **Color Balance & Levels (Post-Stretched)**: Real-time adjustments of *Red Shift*, *Green Shift*, *Blue Shift*, *Brightness*, and *Contrast* to calibrate colors and enhance faint details.
* **Cleaning Filter (Denoise/Smooth)**: Dedicated slider to apply a dynamic Gaussian kernel filter to smooth background noise.
* **Autostretch & Histogram Curve**: Automatic logarithmic stretching (based on median absolute deviation) for raw unstretched images.
* **Color Sampling**: Interactive black level (sky background) and white level (hot white stars) calibration directly on screen.

### 2. Celestial Object Marking & Annotations
* **Custom Annotations**: Click any object to place a precision crosshair (open in the center) and add customized notes.
* **J2000 Coordinates**: Automatically converts pixel coordinates using the WCS model and appends the J2000 RA/Dec string to the annotation.
* **Asteroid Tracking**: Automatically queries the IMCCE SkyBoT database to draw **orange diamonds** at the positions of asteroids with $Mag < 21$ in the viewport.

### 3. Photometry & Magnitude Calculation (Split Workflow)
* **Calibration vs. Measurement separation**:
  * **Mark Calib Stars**: Marks and adds calibration stars (cyan circles) while ignoring variables. Clicking an already selected star deselects/removes it (toggling behavior).
  * **Measure Target**: Calculates the magnitude of any clicked star (including variables and catalog stars).
* **Auto-Calibration Prompt**: Shows a quick pop-up after downloading catalog stars to immediately trigger the automated zero-point calibration.
* **Instrumental Color Index Interpolation (Color FITS)**: For RGB FITS, the program measures the Red and Blue fluxes to calculate:
  $$CI_{\text{instr}} = -2.5 \log_{10}(F_B / F_R)$$
  It uses the calibration stars to fit a color correction line and estimate the physical magnitude of unknown transients silently (no dialog shown for known catalog stars).
* **Target Classification Likelihood**: Estimates and displays the probability percentage of the object type (*Nova, Supernova Ia, Asteroid/Yellow Dwarf, Red Dwarf, Blue Giant*) based on its calculated color index.

### 4. DSS Comparison with Cross-Fade
* **DSS Reference Sky Overlay**: Downloads and caches DSS2 plates from STScI in background.
* **Fade Slider**: Real-time opacity slider (Cross-Fade) between your calibrated image and the reference DSS sky, aligned using WCS.

### 5. Plate Solving & WCS Header Writing
* **Multiple Solvers**: Supports plate solving via local/Vizier catalogs (astroalign) or online via **Nova.astrometry.net**.
* **WCS Save**: Saves solved coordinate systems back into the FITS headers of the original file.

### 6. Gaia Query (Gaia Search Mode)
* **Gaia Star Search Mode**: Clicking on a star queries the GAIA DR3 database to get magnitude, parallax, proper motion, and colors.
* **Green Box, Red X & Aladin Lite**: Draws a **green square** around the click target, a **red X** at the exact Gaia catalog coordinate, and provides a clickable link to open the area in **Aladin Lite** (interactive web sky map).

---

## Installation & How to Run
1. Install Python 3.8+ on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/qsecofr76/FitsManager.git
   cd FitsManager
   ```
3. Install dependencies:
   ```bash
   python install_requirements.py
   ```
4. Run the application:
   ```bash
   python fits_manager.py
   ```

---

## Windows Release (Ready-to-Use EXE)
For Windows users who prefer to run the application without installing Python or configuring development dependencies, a precompiled standalone release is available.
You can download the packaged ZIP archive directly from the **Releases** section of this GitHub repository.

Once you have downloaded the ZIP file:
1. Extract the contents to any folder of your choice.
2. Run the **`FitsManager.exe`** executable file within the extracted folder.
