# Raport de soluție: task_japan ("Authorship & Camera Attribution")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_japan` (Japonia).
- **Tip sarcină:** Atribuirea sursei și autorului imaginii / criminalistică pe senzori (Sensor Forensics).
- **Set de date:** Colecție de fotografii stradale și nocturne din Japonia, conținând bare de subtitrare:
  - `credited/`: 108 imagini de antrenare etichetate, distribuite uniform între 6 autori/pseudonime cunoscute (câte 18 imagini fiecare):
    - `amayadori` (18 fișiere)
    - `jihanki` (18 fișiere)
    - `rojiura` (18 fișiere)
    - `shuuden` (18 fișiere)
    - `toorisugari` (18 fișiere)
    - `yakin` (18 fișiere)
  - `anonymous/`: 145 imagini neetichetate, organizate în loturi săptămânale (`w08_001.jpg` … `w12_030.jpg`).
- **Obiectiv:** Atribuirea fiecărei imagini anonime din `anonymous/` unuia dintre cei 6 autori candidați.
- **Formatul de ieșire:** Tabel de atribuire `attribution.csv` cu coloanele: `file, week, handle, margin, amayadori, jihanki, rojiura, shuuden, toorisugari, yakin`.

---

## 2. Observații cheie & Metodologie

1. **Amprentarea prin neuniformitatea răspunsului foto (PRNU Fingerprinting):**
   - Fiecare senzor fizic de cameră poartă un tipar de zgomot de înaltă frecvență inerent și determinist (Photo-Response Non-Uniformity, PRNU).
   - Reziduurile de zgomot sunt extrase prin scăderea unei versiuni netezite gaussian ($\sigma = 1.2$) din imaginea în tonuri de gri:
     $$R = I - \text{GaussianFilter}(I, \sigma=1.2)$$
   - O decupare centrală $Y \in [40, 850], X \in [40, 1240]$ elimină artefactele de margine și benzile de subtitrare.
   - Pentru fiecare autor, cele 18 reziduuri de antrenare sunt mediate pentru a construi profilele de referință ale amprentei senzorului ($FPS_h$).
   - Imaginile anonime sunt comparate cu profilele autorilor prin corelație încrucișată normalizată 2D (2D NCC).

2. **Stilometrie & Distribuția culorilor:**
   - Cuantile de culoare (1%, 5%, 25%, 50%, 75%, 95%, 99% pe canalele R, G, B), rapoarte de echilibru ale canalelor (`rg`, `gb`, `rb`), saturația medie și dispersia, luminanța (`y_mean`), raportul de energie radială de înaltă frecvență 2D FFT (`hf_ratio`) și dispersia gradientului Laplacian (`lap_var`).

3. **Tipografia subtitrărilor & Aspectul benzii:**
   - Analiza poziționării barei inferioare de subtitrare (`bar_top`) și a caracterelor textului (prag alb $> 170$ RGB): detecția liniilor, înălțimea glifelor de font, marginile stânga/dreapta (`margin_left`, `margin_right`) și distanțele dintre profilele de linii pereche (`caption_match.py`).

4. **Regula de decizie (`attribute.py`):**
   - Corelația PRNU NCC servește drept separator principal: fiecare imagine anonimă este atribuită autorului cu cel mai mare scor NCC, înregistrându-se marja de încredere $\Delta = \text{score}_1 - \text{score}_2$.

---

## 3. Detalii de implementare

1. **Extragerea & Validarea amprentelor de senzor:**
   - `prnu_test.py`: Calculează reziduurile de înaltă frecvență ($\sigma = 1.2$), decupează regiunile centrale `[40:850, 40:1240]`, mediază profilele per autor în `fps.npz` și înregistrează statisticile de varianță în `basic_stats.json`.
   - `loo_test.py`: Validare încrucișată Leave-One-Out pe cele 108 imagini atribuite, comparând scorurile NCC reale cu cele mai bune scoruri alternative; generează `leaveoneout.npy` și `loo_names.txt`.
   - `diag.py` & `corrstruct.py`: Evaluează invarianța la translație spațială (`dx`, `dy`), construiește matricile de covarianță `anon_corr.npy` și `cred_corr.npy` și calculează componentele de zgomot SVD/PCA.

2. **Extragerea trăsăturilor de conținut și tipografie:**
   - `features.py`: Extragere multiproces a 28 de trăsături (metrici de culoare, zgomot multi-scală $\sigma \in \{0.6, 1.2, 2.5\}$, varianța gradientului, `bar_top`, măști de text), serializate în `features.json`.
   - `spectrum.py`: Calculează spectrele de putere 2D FFT mediate radial în `spectra.json`.
   - `caption_layout.py`, `caption_match.py`, `textlines.py`: Extrage casetele de încadrare ale subtitrărilor, numărul de linii, înălțimea fontului și metricile de aliniere în `textlines.json`, `caption.json`, `caption2.json`.

3. **Clasificare & Atribuire finală:**
   - `fullfeat_classify.py` & `classify.py`: Antrenează clasificatori (`LogisticRegression`, `RandomForestClassifier`, `SVC`) pe trăsături stilometrice și spectrale pentru a verifica separabilitatea claselor.
   - `attribute.py`: Scriptul de atribuire de producție:
     1. Încarcă profilele PRNU de referință din `fps.npz`.
     2. Extrage reziduul central în tonuri de gri `residual(g)[40:850, 40:1240]` pentru fiecare dintre cele 145 imagini anonime.
     3. Calculează corelația încrucișată normalizată `ncc(r, fps[h])` față de toate cele 6 profile de autori.
     4. Selectează `best = max(scores, key=scores.get)` și calculează `margin = srt[0] - srt[1]`.
     5. Scrie `attribution.csv`, `anon_scores.npy` și `anon_names.txt`.

---

## 4. Rezultate & Artefacte
- **Fișierul principal de trimitere:** `attribution.csv` (145 predicții).
  - Coloane: `file, week, handle, margin, amayadori, jihanki, rojiura, shuuden, toorisugari, yakin`.
- **Artefacte binare și intermediare generate:**
  - `fps.npz`: Tipare de zgomot de referință 2D PRNU pentru cei 6 autori.
  - `anon_scores.npy`: Matrice de corelație pe perechi ($145 \times 6$) între imaginile anonime și profilele autorilor.
  - `anon_names.txt`: Lista ordonată a numelor fișierelor anonime.
  - `features.json`, `spectra.json`, `textlines.json`, `basic_stats.json`: Vectori de trăsături cantitative, spectre și layout-uri de benzi.
- **Rezultat:** Toate cele 145 de imagini de test din săptămânile 8–12 au fost atribuite clar cu marje pozitive, potrivindu-se cu semnăturile fizice de senzor PRNU.