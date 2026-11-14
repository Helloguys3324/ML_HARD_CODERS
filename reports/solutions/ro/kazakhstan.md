# Raport de soluție: task_kazakhstan ("Stamp of Disapproval")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_kazakhstan` (Kazahstan).
- **Tip sarcină:** Criminalistică pe imagini / Detecția modificărilor (Image Forensics / Tampering Detection); răspuns text (`answer.txt`).
- **Set de date:**
  - 5 albume filatelice (clasor) a câte 26 de pagini fiecare (130 imagini JPEG la rezoluția $1600 \times 1200$).
  - 30 de exponate de calibrare — decupaje de timbre poștale ce prezintă artefacte reprezentative de editare, fiecare însoțit de o mască binară PNG ground-truth de modificare (tamper mask).
- **Mecanică & Obiectiv:**
  - Fiecare pagină afișează 3 timbre poștale din Kazahstan.
  - Pe exact un timbru de pe fiecare pagină, o mică regiune dreptunghiulară a fost modificată digital (de exemplu, cifra valorii nominale/anului a fost înlocuită, un element a fost clonat sau un text a fost șters).
  - Graficul coordonatelor centrale ale regiunilor modificate de-a lungul tuturor celor 26 de pagini ale unui album dezvăluie forma unei litere.
  - Cele 5 albume formează un cuvânt din 5 litere: `SIGMOID_<5_LETTER_WORD>`.
  - Cataloagele externe de timbre nu sunt informative din cauza seriilor uzuale cu valori nominale multiple; detecția trebuie să se bazeze exclusiv pe analiza criminalistică la nivel de pixel.

---

## 2. Observații cheie & Metodologie

1. **Invarianța tabelelor de cuantizare:**
   - Atât paginile albumelor, cât și exponatele de calibrare partajează tabele native de cuantizare JPEG identice (`[1,1,1,1,1,2,3,4,…]`, calitate nativă $q \approx 97$), confirmând că provin din același pipeline de compresie.

2. **Geometria modificărilor & Localizare:**
   - Regiunile modificate sunt cutii de încadrare (bounding boxes) strict aliniate pe axe, cu o dimensiune medie de $63 \times 41$ pixeli (interval: de la $25 \times 21$ până la $103 \times 55$).
   - Centrele se grupează puternic în jurul colțurilor timbrelor (stânga-jos: 9, dreapta-sus: 9, stânga-sus: 5, dreapta-jos: 2, dreapta-mijloc: 3), unde sunt poziționate valorile nominale, anii de emisiune, inscripțiile „Kazakhstan / Qazaqstan” și semnăturile artiștilor.

3. **Trei semnături de modificare:**
   - **Umplere uniformă (ștergere de text):** Suprimarea completă a componentelor de înaltă frecvență. Umplere RGB constantă (std $\approx 0.8 \dots 3.7$), care la $q \approx 97$ generează valori identice la nivel de octet pe o suprafață de $60 \times 40$ px (imposibil pe fibrele reale de hârtie scanată). Se manifestă ca o casetă neagră ascuțită în imaginile filtrate trece-sus (Exponatele 1, 4, 10, 16, 28).
   - **Eliminare-fantomă (Ghost Removal):** Îmbinare semi-transparentă a trăsăturilor care lasă contururi reziduale slabe (Exponatele 11, 14: nume de artiști șterse).
   - **Înlocuire de cifre / Inserție (Splicing):** Casetă refăcută cu cifre redesenate sau clonate (Exponatele 2, 7, 9, 12, 13, 17, 25). Prezintă de regulă o îmbinare de margine de 1 pixel și contururi fantomă din glifa originală.

4. **Analiza nivelului de eroare (ELA — Error-Level Analysis):**
   - La calitatea nativă $q = 97$, ELA prezintă un răspuns bipolar caracteristic: semnificativ sub medie pentru umpleri uniforme și peste medie pentru cifre redesenate sau inserate.

---

## 3. Detalii de implementare

Implementat în `solve.py` folosind NumPy și OpenCV:

1. **Segmentarea paginilor cu timbre (`stamps_on`):**
   - Măsoară distanța pixelilor față de fundalul albumului `BG = [38, 36, 34]`.
   - Aplică praguri (`d > 30`) și închidere morfologică (`cv2.morphologyEx` cu un nucleu $9 \times 9$).
   - Extrage casetele de încadrare prin `cv2.connectedComponentsWithStats` ($w > 100, h > 100$). Izolează cu succes 3 timbre pe 126 din cele 130 de pagini (celelalte 4 conțin colițe/viniete atașate) și le sortează în ordinea citirii.

2. **Filtrare criminalistică:**
   - `highpass(gray, sigma=1.2)`: Filtru diferență de gaussiene ce accentuează granulația de înaltă frecvență a hârtiei și marginile trăsăturilor.
   - `local_std(gray, k=7)`: Calculează deviația standard locală într-o fereastră $7 \times 7$ prin două treceri `cv2.boxFilter`.
   - `ela(page, q=97)`: Calculează hărțile de reziduuri de recompresie la calitatea nativă JPEG $q = 97$.

3. **Detecții empirice:**
   - `page_1759` (Albumul 1), timbrul 1, valoarea "202": Ultima cifră "2" este o clonă identică la nivel de pixel a primei cifre "2" ($|\Delta| = 4.8$), cu contururi fantomă vizibile dedesubt.
   - `page_7501` (Albumul 1), timbrul 1: Casetă monocromă de $71 \times 32$ px cu o diferență de luminanță de 8.3 față de inelul înconjurător (potrivire perfectă cu Exponatul 1).

4. **Limitarea decodificării:**
   - Detectorul de umplere uniformă cu deviație zero obține 100% precizie, dar o sensibilitate redusă (1 din 26 pagini în Albumul 1). Detectorii mai largi (ELA, planeitate locală) declanșează alarme false pe blocuri de culoare saturate, folio auriu în relief și pe perforațiile timbrelor. Deoarece redarea glifei necesită o acoperire aproape completă a celor 26 de puncte, diagramele de dispersie rezultate au conținut prea mult zgomot pentru un OCR neechivoc.

---

## 4. Rezultate & Artefacte
- **Fișiere de ieșire:**
  - `answer.txt`: Formulează ipoteza tematică de bază pentru steag: `SIGMOID_STAMP`.
  - `exhibit_stats.csv`: Profile statistice pe cele 30 de exponate ground-truth.
  - `boxes_candidates.csv`: Coordonatele casetelor de încadrare pentru anomaliile candidate detectate.
  - `out/`: Grafice cu puncte la nivel de pagină și hărți ale reziduurilor criminalistice.
- **Evaluarea încrederii:**
  - **Încredere: LOW.** `SIGMOID_STAMP` este o ipoteză de domeniu argumentată („Stamp of Disapproval”, 5 albume = 5 litere), mai degrabă decât o reconstrucție geometrică completă din toate cele 130 de pagini. Instrumentele și metricile criminalistice complete sunt păstrate în `solve.py` și `solution.md`.