# FitsManager

FitsManager è un'applicazione desktop leggera per Windows dedicata alla visualizzazione, all'analisi rapida e alla calibrazione di base dei file astronomici in formato FITS (Flexible Image Transport System). È stata sviluppata in Python con un'interfaccia grafica basata su Tkinter ed integra le funzionalità WCS (World Coordinate System) per la lettura, conversione e allineamento delle coordinate celesti.

## Caratteristiche principali

* **Visualizzazione ed Istogramma**: Caricamento di file FITS singoli o multicanale (2D/3D), disegno in tempo reale degli istogrammi separati per i canali Rosso, Verde, Blu e RGB/Luminanza.
* **Debayering automatico o manuale**: Supporto per i pattern Bayer principali (RGGB, BGGR, GRBG, GBRG) per ricostruire i colori reali dai sensori a matrice di colore (CFA) a partire da dati a 16 bit.
* **Correzione Curve e Gamma**: Manipolazione interattiva delle curve di livello canale per canale basata su interpolazione spline cubica monotonica. Funzione di autostretch logaritmico basata sulla deviazione standard mediana per evidenziare dettagli deboli su immagini non stirate.
* **Bilanciamento del Colore sui Campioni (Nero/Bianco)**:
  * **Sky Background (Nero)**: Calibrazione del fondo cielo su un'area $10\times10$ pixel. L'algoritmo filtra i pixel saturi (>150% della media) per escludere stelle e allinea i canali ad un livello di grigio scuro neutro di riferimento (~3%).
  * **White Star (Bianco)**: Allineamento del punto di bianco prendendo in esame la parte più luminosa della stella campionata (escludendo il nucleo saturo). Preserva una frazione (15%) della naturale tendenza calda/fredda della stella per non alterare la fisica cromatica.
* **Annotazione ed Esportazione**:
  * Annotazione interattiva con crocicchio verde brillante aperto al centro (per non coprire il nucleo della stella).
  * Possibilità opzionale di convertire la coordinata del pixel cliccato tramite il WCS dell'header e memorizzare/esportare in coda al testo il valore di coordinate in **J2000**.
  * Esportazione finale dell'immagine stirata e calibrata in formato PNG, JPG o TIFF con le annotazioni impresse ad alta visibilità.
* **Navigazione ed Viewport Cropping (Prestazioni 60 FPS)**:
  * Zoom fluido bidirezionale mirato verso la posizione del cursore del mouse con debouncing a bassa risoluzione durante il movimento per massima fluidità.
  * Spostamento (Pan) tramite trascinamento con tasto destro (cursore a manina) e barre di scorrimento.
  * **Viewport Crop Optimization**: Per evitare calcoli su file giganteschi (es. 26 Megapixel da ASI2600), l'applicazione ritaglia e riscala solo la porzione visibile sul monitor. Il warping WCS e il blend con il cielo di riferimento avvengono istantaneamente in soli ~3ms.
  * Orientamento telescopio: Flip Horizontal e Flip Vertical coerenti con le letture WCS e il piazzamento delle annotazioni.
* **DSS Reference Sky Overlay (Download in Background)**:
  * Download automatico di lastre reali DSS2 (red plate) tramite query CGI HTTP a STScI.
  * Lettura delle intestazioni WCS native dei file DSS scaricati per un allineamento e cross-fade (Blend) millimetrico con la tua foto tramite OpenCV.
  * Scorrimento in background multi-thread con finestra di monitoraggio, barra di avanzamento e pulsante per annullare l'operazione.
  * **Cache Intelligente**: Scansione istantanea e caricamento all'avvio dei tasselli adiacenti e sovrapposti già salvati localmente nella cartella `dss_cache/` (con tolleranza incrementata a 10 arcominuti per evitare download duplicati). I tasselli mancanti vengono scaricati in background.
* **Integrazione MPC Asteroidi (IMCCE SkyBoT)**:
  * Query automatica al database IMCCE SkyBoT per scaricare ed identificare la posizione di tutti gli asteroidi e comete con magnitudine visiva $Mag < 21$ presenti nell'inquadratura al momento preciso dello scatto (calcolato tramite la chiave `DATE-OBS` dell'header FITS).
  * Rappresentazione grafica degli asteroidi identificati tramite **rombi arancioni** riportanti nome dell'oggetto e magnitudine.
* **Fotomeria e Calibrazione Zero-Point**:
  * Download automatico dei cataloghi stellari GAIA o APASS DR9 per la calibrazione fotometrica dell'immagine.
  * Misurazione del flusso stellare tramite aperture photometry con centroiding automatico delle coordinate. Calcolo del punto di zero (Zero-Point) medio e stima/annotazione della magnitudine reale di qualsiasi stella sconosciuta selezionata.
* **Modalità Bianco e Nero**: Toggle rapido per visualizzare l'immagine in scala di grigi mantenendo le annotazioni in verde brillante.
* **Cronologia**: Stack completo di Undo e Redo (`Ctrl+Z` / `Ctrl+Y`).
* **Drag & Drop**: Possibilità di aprire i file FITS trascinandoli direttamente sull'interfaccia.

## Requisiti

L'applicazione richiede Python 3.8+ e le seguenti librerie:
* `numpy`
* `opencv-python`
* `pillow`
* `astropy`
* `scipy`
* `tkinterdnd2`

## Installazione ed Avvio da sorgente

1. Clona la repository:
   ```bash
   git clone https://github.com/qsecofr76/FitsManager.git
   cd FitsManager
   ```

2. Installa le dipendenze utilizzando lo script fornito:
   ```bash
   python install_requirements.py
   ```
   *(Lo script aggiornerà pip ed installerà i pacchetti richiesti tramite setup di librerie precompilate).*

3. Avvia l'applicazione:
   ```bash
   python fits_manager.py
   ```

## Compilazione dell'eseguibile autonomo (EXE)

Per generare un pacchetto auto-consistente distribuibile senza installare Python:
```bash
pip install pyinstaller
pyinstaller -y FitsManager.spec
```
L'eseguibile compilato (`FitsManager.exe`) e tutte le sue librerie collegate si troveranno nella cartella `dist/FitsManager`.
