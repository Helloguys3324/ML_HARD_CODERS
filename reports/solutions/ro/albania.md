# Raport de Soluție: task_albania („Looks like an anomaly!”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_albania` (Albania).
- **Dificultate și Scor:** Medie, 200 / 200 puncte.
- **Domeniu / Track:** Detecția Anomaliilor în Serii de Timp (Time-Series Anomaly Detection), Streaming ML.
- **Context:** Monitorizarea stării structurale și de mediu a unui buncăr subteran din beton din Albania prin intermediul senzorilor integrați.
- **Date de Intrare:** Citiri de senzori în flux continuu (`train-input.csv`) ingerate strict pas cu pas (`update(reading: dict)`). Șase caracteristici numerice: `temperature_c`, `humidity_pct`, `vibration_g`, `acoustic_db`, `air_pressure_kpa`, `co2_ppm`, alături de metadatele `timestamp` și `bunker_zone`.
- **Tipuri de Anomalii:**
  1. *Anomalii punctuale (Point anomalies)* (valori aberante bruște, instantanee);
  2. *Derive graduale (Gradual drifts)* (deplasări lente ale nivelului de bază);
  3. *Anomalii colective (Collective anomalies)* (deviații moderate concurente pe caracteristici corelate).
- **Constrângeri:** Procesare în timp real (fără privire înainte / no lookahead, fără acces la rețea sau fișiere externe), mediu de execuție Python standard (`numpy`, `pandas`, `scikit-learn`), execuție pe CPU sub 60 de secunde.
- **Metrică de Evaluare:**
  Scoruri de anomalie $\in [0.0, 1.0]$ pragificate la $0.50$.
  $$\text{Score} = 100 \times (0.80 \times \text{overall\_F1} + 0.05 \times \text{point\_recall} + 0.075 \times \text{drift\_recall} + 0.075 \times \text{collective\_recall})$$
  Prag de promovare: $\text{Score} \ge 70.00 / 100$.

---

## 2. Perspective Cheie și Metodologie
Soluția folosește un cadru statistic de streaming robust și ușor, fără overhead-ul rețelelor neuronale:
1. **Arhitectură cu Fereastră Glisantă Dublă (Dual Sliding-Window):**
   - **Fereastră Lungă ($W_{\text{long}} = 720$ pași):** Calculează statistici de bază robuste folosind mediana mobilă și Deviația Mediană Absolută ($\text{MAD}$, cu scală normală $\sigma \approx 1.4826 \times \text{MAD}$).
   - **Fereastră Scurtă ($W_{\text{short}} = 30$ pași):** Urmărește schimbările locale instantanee de nivel pentru detecția rapidă a derivei (drift).
2. **Prag Minim de Varianță (`SIGMA_FLOOR`):** Protejează împotriva împărțirii la zero și a varianței degenerate în timpul fazei de încălzire (warm-up) sau în regimuri cu zgomot redus (`temperature_c: 0.10`, `humidity_pct: 0.20`, `vibration_g: 0.003`, `acoustic_db: 0.10`, `air_pressure_kpa: 0.02`, `co2_ppm: 3.0`).
3. **Agregarea Deviațiilor Multiaspectuale:**
   - Evaluează deviația instantanee $z_{\text{inst}}[f] = \frac{x_f - \text{med}_f}{\sigma_f}$ și deplasarea nivelului pe termen scurt $z_{\text{level}}[f] = \frac{\text{median}(\text{short}_f) - \text{med}_f}{\sigma_f}$ înainte de a adăuga noul punct (prevenind scurgerea de date / self-leakage).
   - Combină maximele pe caracteristici individuale ($spike = \max_f |z_{\text{inst}}[f]|$, $lev = \max_f |z_{\text{level}}[f]|$) cu media pătratică a primelor 3 caracteristici deviate ($multi = \sqrt{\frac{1}{3}\sum_{k=1}^3 (top_k(event))^2}$) pentru a surprinde anomaliile colective.
4. **Mapare Calibrată a Probabilităților:**
   Mapează deviația brută $raw = \max(spike, lev, multi)$ în $[0.0, 1.0]$ printr-un CDF exponențial:
   $$\text{score} = 1.0 - \exp\left(-\frac{\max(raw, 0)}{4.3}\right)$$
   Aceasta aliniază limita de decizie de $0.50$ cu o deviație de aproximativ $3.0\sigma$.

---

## 3. Detalii de Implementare Pas cu Pas

### Scripturi Principale (`task_albania/`)
- **`detector.py` (Modul de Trimitere):**
  - Implementează `Detector.update(self, reading: dict) -> float`.
  - Păstrează istoricul într-un buffer folosind `collections.deque(maxlen=720)` și `deque(maxlen=30)`.
  - Returnează `0.0` în timpul fazei inițiale de încălzire ($n < 40$).
  - Evaluează deviațiile, actualizează bufferele circulare, calculează metrica compusă $raw$ și aplică maparea trunchiată a probabilității.
- **`solve.py` (Harness Local de Evaluare și Verificare):**
  - Simulează ingestia în flux continuu din `train-input.csv`.
  - Calculează matricea de confuzie ($TP, FP, FN, TN$), precizia, rapelul (recall), scorul global $F_1$ și rapelurile specifice (`r_point`, `r_drift`, `r_coll`).
  - Verifică dacă scorul final al competiției depășește $70.00$.

---

## 4. Rezultate și Artefacte
- **Stare:** `PASSED` (Scor: **200 / 200** puncte).
- **Tentativă de Trimitere:** Tentativa 1 (`2026-11-14T11:20:37.696Z`, trimis de: `Veaceslav C`).
- **Performanță de Detecție:**
  - Robust împotriva vârfurilor punctuale, derivelor lente și evenimentelor colective pe mai mulți senzori.
  - Zero pozitive false în timpul ferestrei inițiale de inițializare.
- **Flag:**
  ```text
  SIGMOID_BUNKER
  ```
- **Artefacte Cheie:**
  - `detector.py` (4.287 octeți, clasă autonomă de detector);
  - `solve.py` (harness de verificare pentru streaming).
