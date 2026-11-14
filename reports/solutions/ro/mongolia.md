# Raport de soluție: task_mongolia ("The Baa-rilliant task!")

## 1. Prezentare generală & Obiectiv
- **ID sarcină:** `task_mongolia` (Mongolia).
- **Dificultate & Punctaj:** Medie (200 puncte); trimitere prin răspuns text (`answer.txt`).
- **Set de date:**
  - Set de antrenare: 280 imagini aeriene din dronă ale unor pășuni (`train/train_000.jpg` … `train_279.jpg`), $600 \times 600$ RGB.
  - `train_counts.csv`: Numărători reale (ground-truth) de oi per cadru de antrenare (de la 0 la 119, medie 41.5).
  - `train_points.csv`: Coordonate la nivel de pixel ale centrelor oilor (11.629 puncte adnotate; sumele corespund exact cu `train_counts.csv`).
  - Set de test: 333 imagini aeriene (`survey/survey_0001.jpg` … `survey_0333.jpg`), $600 \times 600$.
  - `survey_plots.csv`: Jurnal de telemetrie de zbor cu coordonate spațiale centrale în metri (`image, x_m, y_m`).
- **Obiectiv:** Numărarea oilor pe toate cele 333 parcele survolate, reconstrucția topologiei grilei de zbor și decodificarea cuvântului-cheie/steagului ascuns în formatul `SIGMOID_<WORD>`.

---

## 2. Observații cheie & Metodologie

1. **Reconstrucția & Geometria grilei de zbor:**
   - Coordonatele $(x, y)$ din `survey_plots.csv` urmează o grilă dreptunghiulară regulată, supusă unor ușoare deviații GPS.
   - Analiza deplasărilor celor mai apropiați vecini prin `scipy.spatial.cKDTree` ($r < 52$ m) a identificat doi vectori de bază ortogonali:
     $$u = (44.92, 22.36)\text{ m}, \quad v = (22.46, -44.19)\text{ m} \quad (|u| \approx 50.2\text{ m}, |v| \approx 49.8\text{ m})$$
     corespunzând unei grile ortogonale de 50 de metri rotite cu $\approx 26.6^\circ$.
   - Descompunerea coordonatelor parcelelor în această bază cu rotunjire întreagă $(i, j)$ produce o grilă exactă de $37 \times 9$:
     - Exact 333 parcele, zero coliziuni, fiecare celulă ocupată o singură dată (reziduu $< 0.37$).
   - Dimensiunea de $37 \times 9$ sugerează puternic un cuvânt din 6 litere redat într-un font matriceal standard de $5 \times 7$:
     $$37 = 6 \times 5\text{ (lățime glifă)} + 5\text{ (spațiere litere)} + 2\text{ (margini orizontale)}$$
     $$9 = 7\text{ (înălțime glifă)} + 2\text{ (margini verticale)}$$

2. **Numărare clasică de oi prin ML (Fair-Play):**
   - S-a proiectat un ansamblu ușor din două modele, antrenate de la zero, fără modele pre-antrenate externe:
     1. **Regresie globală pe cadru (`global_reg.py`):** 46 de trăsături globale incluzând rapoarte de alb $V \cdot (1 - S)^{1.2}$ pe 5 praguri, statistici ale canalelor HSV/Lab, filtre morfologice top-hat albe pe 3 scale, numărători de vârfuri Difference-of-Gaussians (DoG) pe 6 praguri și filtre top-hat negre pentru oi închise la culoare. 5-fold CV: **MAE = 8.2 oi, Pearson $r = 0.906$**.
     2. **Regresie de densitate locală a celulelor (`cell_density.py` + `cell_reg_fast.py`):** Împarte fiecare cadru într-o grilă $10 \times 10$ de petice de $60 \times 60$ px. Extrage 21 de trăsături locale per petic; antrenează LightGBM pentru a prezice numărul de oi la nivel de celulă. Însumarea tuturor celor 100 de celule dă numărul total per cadru (**MAE = 7.5**).
     3. **Ansamblu:** Combină predicțiile prin formula $\text{count} = \text{round}(0.5 \cdot \text{pred}_{\text{global}} + 0.5 \cdot \text{pred}_{\text{cell}})$. Total estimat de oi pe cele 333 parcele: $\approx 11.693$ (medie 35.1, mediană 33, interval 3 până la 132).

3. **Limitarea decodificării:**
   - S-a evaluat matricea de numărători $37 \times 9$ față de fonturi raster $5 \times 7$ pe 8 grupuri de simetrie (`font_decode.py`, `symmetry_decode.py`), eliminare a tendinței spațiale 2D (detrending), aplicare de praguri pe percentile și encodări alternative.
   - Zgomotul de regresie (MAE $\approx 8$) și contracția spre medie estompează marginile glifelor, împiedicând un OCR automat neechivoc din punct de vedere statistic.

---

## 3. Detalii de implementare

1. **Orchestrarea grilei & a pipeline-ului (`solve.py`):**
   - `fit_lattice(csv_path)`: Construiește arborele spațial k-d, grupează vectorii vecini, rezolvă baza $(u, v)$, mapează coordonatele la $(i, j)$ discrete și salvează `survey_plots_ij.csv`.
   - Gestionează extragerea trăsăturilor și îmbină predicțiile în `counts.csv`.

2. **Antrenarea modelelor & Predicție:**
   - `global_reg.py`: Extrage 46 de trăsături globale de culoare și morfologie; generează `survey_pred_global.csv`.
   - `cell_density.py` & `cell_reg_fast.py`: Extrage trăsături la nivel de petic și rulează o regresie rapidă, generând `survey_pred_cell.csv`.
   - `make_counts.py`: Îmbină și formatează predicțiile.

3. **Module de decodificare & vizualizare:**
   - `font_decode.py` & `symmetry_decode.py`: Comparator de fonturi raster $5 \times 7$ pe diverse aliniamente spațiale.
   - `render_grid.py`, `explore_grid.py`: Randează hărți termice 2D în tonuri de gri ale distribuțiilor de densitate.

---

## 4. Rezultate & Artefacte
- **Tabele de ieșire:**
  - `counts.csv`: Numărul estimat de oi pentru toate cele 333 parcele (`image, count`).
  - `survey_plots_ij.csv`: Maparea imaginilor din zbor pe coordonatele grilei $37 \times 9$.
  - `survey_pred_global.csv`, `survey_pred_cell.csv`: Predicțiile individuale ale modelelor.
- **Ipoteza steagului & Evaluare:**
  - `answer.txt`: Conține ipoteza de domeniu `SIGMOID_SHEEP`.
  - **Încredere: LOW.** Glifele nu au putut fi decodificate peste nivelul de zgomot al regresiei.
  - Candidați plauzibili identificați:
    - `SIGMOID_CENSUS` (potrivire exactă de 6 litere pentru aspectul $37 \times 9$, reprezentând un recensământ al oilor).
    - `SIGMOID_REPORT` (corespunde unor potriviri slabe de font `REP…`).
    - `SIGMOID_SUREG` (în mongolă *süreg* = turmă).