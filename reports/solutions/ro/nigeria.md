# Raport de Soluție: task_nigeria ("The Gángan Never Lies")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_nigeria` (Nigeria).
- **Dificultate și Scor:** Easy, 100 / 100 puncte.
- **Domeniu:** General AI / Criminalistică Audio și Procesarea Semnalelor.
- **Format Ieșire:** Flag secret `SIGMOID_<WORD>` în `answer.txt`.
- **Premisă:**
  - Un proiect de digitalizare a arhivelor din Lagos a scos la iveală fișierul `challenge.wav`, înregistrat de un inginer de sunet pensionat, prezentând o tobă vorbitoare Yoruba (*gángan*).
  - Inginerul a ascuns un mesaj secret în interiorul înregistrării, care trebuie extras pentru a recupera flag-ul.

---

## 2. Perspective Cheie și Metodologie Criminalistică
- **Triage-ul Containerului și al Metadatelor:**
  - Inspecția antetului RIFF/WAVE a confirmat o structură curată conținând doar două chunk-uri standard: `fmt ` (16 octeți) și `data` (448,512 octeți), fără octeți finali adăugați (trailing bytes) sau chunk-uri străine de metadate (`LIST`, `INFO`).
  - Specificații audio: Mono, 16-bit signed LE, frecvență de eșantionare $f_s = 22,050\text{ Hz}$, 224,256 eșantioane (durată: 10.17 s).
- **Verificări Negative de Steganografie:**
  - Analiza LSB pe toate cele 16 planuri de biți a evidențiat distribuții nepolarizate ale biților (valori medii 0.4988–0.5013).
  - Scanările automate pentru încărcături ASCII înglobate, tonuri DTMF și cod Morse au fost negative.
- **Analiza Frecvență-Timp a Spectrogramei:**
  - Inspecția frecvențelor a identificat un tipar modulat în amplitudine artificial în banda **3.0–3.7 kHz**, afișând forme vizuale de litere.
  - **STFT de Înaltă Rezoluție:** Transformata Fourier pe Timp Scurt (Short-Time Fourier Transform) folosind o fereastră Hann cu `nperseg = 1024` și `noverlap = 1008` (rezoluție temporală $\approx 0.73\text{ ms}$).
  - **Albire Spectrală (Spectral Whitening):** S-a scăzut mediana temporală din fiecare rând de frecvență (`mag - np.median(mag, axis=1, keepdims=True)`) pentru a elimina energia acustică de fundal a tobei.
  - Amplitudinile au fost normalizate și s-a aplicat binarizare pe bază de prag pentru a obține glife clare.

---

## 3. Detalii de Implementare
Fluxul de extracție este implementat în `solve.py`, `solution.md` și scripturile din `exploration/`:

1. **Decodarea Audio (`read_wav` în `solve.py`):**
   - Citește eșantioanele PCM brute din `challenge.wav` și verifică integritatea antetului.
2. **Generarea și Filtrarea Spectrogramei:**
   - Calculează `scipy.signal.stft(x, fs=22050, window='hann', nperseg=1024, noverlap=1008)`.
   - Extrage secțiunea matricei de frecvențe între 2.8 kHz și 3.9 kHz.
   - Efectuează albirea rândurilor, restrânge valorile negative la zero și normalizează intervalul dinamic la $[0, 1]$.
3. **Decodarea Glifelor și Verificarea:**
   - Între $t = 0.70\text{ s}$ și $t = 9.65\text{ s}$, sunt redate clar 11 majuscule egal distanțate:
     $$\text{T}\quad\text{A}\quad\text{L}\quad\text{K}\quad\text{I}\quad\text{N}\quad\text{G}\quad\text{D}\quad\text{R}\quad\text{U}\quad\text{M}$$
   - Formează cuvântul compus englezesc `TALKINGDRUM` (traducerea directă a *gángan*).
   - Analiza fișierului audio inversat temporal (`spectrogram_reversed.png`) a confirmat redarea direcțională normală (`MURDGNIKLAT`).
   - Spațierea dintre `G` și `D` este identică cu spațierea dintre celelalte caractere, excluzând caracterele underscore sau delimitatorii.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 100 / 100 puncte
- **Flag Recuperat:** `SIGMOID_TALKINGDRUM`
- **Artefacte Rezultate:**
  - `answer.txt` — conține șirul de trimitere a flag-ului;
  - `spectrogram.png` — vizualizarea spectrului pe întregul interval;
  - `spectrogram_message.png` — banda de frecvență izolată a mesajului;
  - `message_binary2.png` — mască binarizată de text pentru verificare vizuală/OCR.
