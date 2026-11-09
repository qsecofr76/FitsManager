# FitsManager

FitsManager è un'applicazione desktop leggera per Windows dedicata alla visualizzazione, all'analisi rapida, alla calibrazione e alla misurazione fotometrica di file astronomici in formato FITS (Flexible Image Transport System). Integra funzioni avanzate di allineamento e coordinate celesti (WCS), interrogazioni multi-catalogo stellare ed extragalattico, download di survey astronomiche reali (DSS, Pan-STARRS, Legacy Survey DR9/DR10), calibrazione fotometrica ed estrazione di dati fisici in tempo reale.

*(English version below)*

---

## Novità ed Evoluzione (v1.2.0)

La versione **1.2.0** introduce rilevanti potenziamenti su affidabilità dei cataloghi, identificazione extragalattica e survey astronomiche di riferimento:

1. **Integrazione Avanzata NASA/IPAC NED & Classificazione Galassie**:
   - Interrogazione automatica REST delle nuove API di NED (*NASA/IPAC Extragalactic Database*).
   - Identificazione scientifica rigorosa tra stelle galattiche e oggetti extragalattici (basata su tipo fisico, redshift cosmologico $z$, velocità radiale $cz$ e cataloghi storici NGC, IC, UGC, PGC, 2MASX).
   - Riconoscimento ed esclusione automatica di sorgenti puntiformi infrarosse generiche (`IrS`, `WISEA`, `2MASS`) per evitare falsi positivi.
   - Visualizzazione grafica con **cerchietto azzurro (`#38bdf8`)** ed etichetta dell'oggetto extragalattico sulla mappa e sulle immagini esportate.
2. **Supporto Ufficiale NASA IRSA ed ESA TAP per Gaia DR3**:
   - Passaggio a **NASA IPAC / IRSA Simple Cone Search** ed **ESA Gaia TAP Server** come provider primari ultra-resilienti, indipendenti dai server europei di Strasburgo/CDS.
   - Estrazione dei modelli DSC (*Discrete Source Classifier*) per stimare con precisione la probabilità astrofisica (Star vs Galaxy vs Quasar).
   - Raggio di tolleranza allargato e ordinamento ponderato per agganciare con precisione il centroide anche da click manuali.
3. **Nuove Survey di Riferimento (Reference Sky Surveys: DSS, Pan-STARRS, Legacy Survey)**:
   - Sottomenu dedicato con gestione separata e dissolvenza (Cross-Fade) su lastre DSS2, tasselli Pan-STARRS e **Legacy Survey (DECaLS / BASS / MzLS)**.
   - Algoritmo a cascata intelligente: *ls-dr10* (`r`/`grz`) per Declinazioni $\le +32^\circ$ e fallback automatico su *ls-dr9-north* per il cielo settentrionale ($> +32^\circ$).
   - Rilevamento automatico e scarto delle zone non coperte o con tile nere (es. piano galattico $|b| < 15^\circ$).
4. **Resilienza e Mirror Failover Automatico per Vizier**:
   - Monitoraggio continuo dello stato dei mirror mondiali (Harvard, Cambridge, Tokyo, Strasburgo).
   - Failover automatico trasparente allo specchio funzionante con persistenza automatica della preferenza in `settings.json`.
5. **Splash Screen Dinamico all'Avvio**:
   - Schermata di caricamento rapida e fluida con icona dell'applicazione e barra di avanzamento, con ottimizzazione dei tempi di import dei moduli pesanti (lazy loading).

---

## Caratteristiche Principali (Italiano)

### 1. Correzione dei Livelli, Bilanciamento del Bianco e Filtri
* **Color Balance & Levels (Post-Stretched)**: Regolazione fine in tempo reale di *Red Shift*, *Green Shift*, *Blue Shift*, *Luminosità* e *Contrasto* per calibrare i colori e far risaltare i dettagli più deboli.
* **Filtro di Pulizia (Denoise/Smooth)**: Slider dedicato per applicare un filtro kernel gaussiano dinamico per ammorbidire il rumore di fondo dell'immagine.
* **Autostretch ed Curve di Livello**: Stretching logaritmico automatico (basato sulla deviazione standard mediana) per immagini grezze non stirate.
* **Campionamento del Colore**: Calibrazione interattiva sul nero (fondo cielo) e sul bianco (stelle calde) selezionando direttamente i pixel di riferimento a schermo.
* **Rimozione Gradienti**: Strumento di sottrazione e correzione del gradiente di luminosità dello sfondo cielo per uniformare il campo dell'immagine.

### 2. Marcatura Oggetti Celesti ed Annotazioni
* **Annotazioni Personalizzate**: Clicca su qualsiasi oggetto per posizionare un mirino di precisione (aperto al centro) inserendo note di testo personalizzate.
* **Integrazione coordinate J2000**: Il programma calcola e allega automaticamente le coordinate celesti RA/Dec J2000 al testo dell'annotazione.
* **Tracciamento Asteroidi**: Interroga in automatico il database IMCCE SkyBoT per visualizzare con **rombi arancioni** la posizione degli asteroidi con $Mag < 21$ presenti nell'inquadratura al momento dello scatto.
* **Ricerca Transienti (TNS)**: Interrogazione diretta del Transient Name Server per contrassegnare novae e supernovae con marker e note ad alta visibilità.

### 3. Fotometria e Calcolo della Magnitudine
* **Separazione tra Calibrazione e Misura**:
  * **Mark Calib Stars**: Permette di marcare e aggiungere le stelle di calibrazione stabili (cerchi azzurri) escludendo le stelle variabili. Cliccando una stella già inserita, questa viene rimossa (toggle).
  * **Measure Target**: Permette di calcolare la magnitudine di qualsiasi stella o sorgente debole.
* **Stima della Magnitudine Limite**: Calcolo algoritmico della sensibilità massima e della magnitudine limite raggiunta dal singolo scatto.
* **Interpolazione del Colore (FIT a Colori)**: Nei FIT RGB, il programma misura il flusso nei canali Rosso e Blu per ricavare l'indice di colore strumentale:
  $$CI_{\text{str}} = -2.5 \log_{10}(F_B / F_R)$$
  Utilizza le stelle di calibrazione per stimare con massima precisione la magnitudine fisica reale di transitori ed oggetti sconosciuti.
* **Classificazione Probabilistica del Target**: Stima percentuale della tipologia di oggetto in base all'indice di colore (*Nova, Supernova Ia, Asteroide, Nana Rossa, Gigante Blu*).

### 4. Survey Celesti di Confronto (DSS, Pan-STARRS, Legacy Survey)
* **Survey Multiple**: Download in background e cache locale per lastre DSS2, skycell Pan-STARRS e tasselli FITS Legacy Survey (DR10 / DR9).
* **Fade Slider**: Slider di opacità in tempo reale (Cross-Fade) tra la tua foto astronomica e la survey ufficiale perfettamente allineata tramite modello astrometrico WCS.

### 5. Plate Solving e Riscrittura Coordinate WCS
* **Plate Solving Flessibile**: Supporta la risoluzione tramite astrometri locali/Vizier (astroalign) o tramite il risolutore online di **Nova.astrometry.net**.
* **Salvataggio WCS**: Riscrive le coordinate risolte direttamente nell'intestazione (Header) del file FITS originale senza perdita di dati.
* **Supporto Immagini Standard (TIFF/PNG/JPG)**: Caricamento e persistenza automatica del WCS e delle annotazioni tramite file sidecar JSON (`.info.json`).

### 6. Modalità Gaia & NED Object Search
* **Analisi Astrofisica Completa**: Cliccando su qualsiasi oggetto in questa modalità, l'applicazione interroga simultaneamente Gaia DR3 e NASA NED, fornendo:
  - Coordinate RA/Dec J2000 esatte, magnitudini $G, BP, RP$, indice di colore $BP-RP$.
  - Parallasse, distanza in anni luce e parsec, moto proprio su entrambi gli assi, velocità radiale.
  - Temperatura effettiva $T_{\text{eff}}$ e classe spettrale stellare (Classi O, B, A, F, G, K, M).
  - Classificazione astrofisica Stella vs Galassia (probabilità Gaia DSC + cross-match NED con redshift $z$).
* **Feedback Grafico Triplo**:
  - **Quadratino Verde**: punto cliccato manualmente.
  - **Croce Rossa ("Gaia Star")**: centroide esatto della stella Gaia.
  - **Cerchietto Azzurro ("NED")**: posizione reale della galassia confermata.
* **Link Diretti Cliccabili**: Collegamenti rapidi ai portali ufficiali **SIMBAD**, **NASA NED** e **Legacy Survey Sky Viewer**.

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
* **Color Sampling**: Interactive black level (sky background) and white level (hot stars) calibration directly on screen.
* **Gradient Removal**: Background gradient estimation and subtraction tool to flatten uneven sky background.

### 2. Celestial Object Marking & Annotations
* **Custom Annotations**: Click any object to place a precision crosshair (open in the center) and add customized notes.
* **J2000 Coordinates**: Automatically converts pixel coordinates using the WCS model and appends the J2000 RA/Dec string to the annotation.
* **Asteroid Tracking**: Automatically queries the IMCCE SkyBoT database to draw **orange diamonds** at the positions of asteroids with $Mag < 21$ in the viewport.
* **Transient Search (TNS)**: Directly queries the Transient Name Server to locate and highlight novae and supernovae.

### 3. Photometry & Magnitude Calculation
* **Calibration vs. Measurement separation**:
  * **Mark Calib Stars**: Marks and adds calibration stars (cyan circles) while ignoring variables (toggling behavior).
  * **Measure Target**: Calculates the magnitude of any clicked star or transient source.
* **Limiting Magnitude Estimation**: Automated detection of the maximum depth and sensitivity reached in the frame.
* **Instrumental Color Index Interpolation (Color FITS)**: For RGB FITS, the program measures the Red and Blue fluxes:
  $$CI_{\text{instr}} = -2.5 \log_{10}(F_B / F_R)$$
  Fits a calibration line using reference stars to determine real physical magnitudes.
* **Target Classification Likelihood**: Probabilistic classification based on measured color index (*Nova, Supernova Ia, Asteroid, Red Dwarf, Blue Giant*).

### 4. Reference Sky Surveys (DSS, Pan-STARRS, Legacy Survey)
* **Multiple Surveys**: Background downloading and caching of DSS2 red plates, Pan-STARRS skycells, and Legacy Survey tiles (DR10 / DR9).
* **Fade Slider**: Real-time opacity slider (Cross-Fade) between your image and reference surveys aligned via WCS.

### 5. Plate Solving & WCS Header Writing
* **Multiple Solvers**: Supports plate solving via local/Vizier catalogs (astroalign) or online via **Nova.astrometry.net**.
* **WCS Header Save**: Saves solved coordinate systems back into the FITS headers of the original file.
* **Standard Images Support (TIFF/PNG/JPG)**: Automatic persistence of WCS and annotations via sidecar JSON files (`.info.json`).

### 6. Gaia DR3 & NASA NED Object Search Mode
* **Multi-Catalog Astrophysical Inspection**: Single-click query querying Gaia DR3 and NASA/IPAC NED simultaneously:
  - Exact J2000 coordinates, $G, BP, RP$ magnitudes, $BP-RP$ color index.
  - Parallax, distance in parsecs and light-years, proper motion, radial velocity.
  - Effective temperature $T_{\text{eff}}$ and stellar spectral class (O, B, A, F, G, K, M).
  - Star vs Galaxy classification (Gaia DSC model probabilities + NED confirmation with redshift $z$ and radial velocity $cz$).
* **Triple Visual Markers**:
  - **Green Box**: manual click position.
  - **Red X ("Gaia Star")**: Gaia star centroid.
  - **Cyan Circle ("NED")**: confirmed extragalactic galaxy position.
* **Direct Official Links**: One-click links to **SIMBAD**, **NASA NED**, and **Legacy Survey Interactive Viewer**.

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
