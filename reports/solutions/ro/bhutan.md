# Raport de Soluție: task_bhutan („The 300-Year Old Tiger Nest Painting”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_bhutan` (Bhutan).
- **Dificultate și Scor:** Medie, 200 / 200 puncte.
- **Domeniu / Track:** Vedere Computațională (Computer Vision), Reconstrucție Puzzle 1D (1D Jigsaw Puzzle Reconstruction), Optimizare Combinatorie.
- **Context:** Restaurarea unei picturi străvechi pe sul de mătase de acum 300 de ani ce ilustrează mănăstirea Paro Taktsang („Cuibul Tigrului”) din Bhutan, tăiată vertical în 192 de fâșii neordonate.
- **Date de Intrare:** Director cu 192 de fâșii de imagine amestecate (`input/` sau `train-input/`) denumite arbitrar fără indicii de poziție.
- **Criterii de Verificare (Imagine Privată Ascunsă):**
  1. **SSIM $\ge 0.90$** (Indexul de Similaritate Structurală față de imaginea originală);
  2. **Order Match Ratio $\ge 0.90$** (fracțiunea de fâșii corect ordonate);
  3. **Timp de Rulare $\le 20.0$ secunde** (limită maximă strictă de execuție).
- **Livrabile:**
  - `main.py` (algoritm autonom de rezolvare a puzzle-ului);
  - `reconstructed_image.png` (imagine recompusă prin îmbinare orizontală);
  - `reconstructed_order.json` (lista ordonată a fișierelor de fâșii).
- **Format Flag:** `SIGMOID_<WORD>`.

---

## 2. Perspective Cheie și Metodologie

Problema se reduce la găsirea unui drum hamiltonian de cost minim peste un graf orientat de afinitate a marginilor $G = (V, E)$:
1. **Matricea de Cost Multiaspectual a Granițelor $C[i, j]$:**
   Plasarea fâșiei $i$ la stânga fâșiei $j$ este evaluată pe baza a trei semnale complementare de continuitate:
   - **Discontinuitate Cromatică la Graniță ($d_{\text{color}}$):** Distanța Manhattan (L1) dintre marginea dreaptă $r_1$ a fâșiei $i$ și marginea stângă $l_0$ a fâșiei $j$.
   - **Aliniere a Gradientului Vertical ($d_{\text{vgrad}}$):** Diferența L1 dintre gradienții verticali $\Delta_y r_1$ și $\Delta_y l_0$, asigurând continuitatea pantelor arhitecturale și a reliefului stâncos.
   - **Continuitate de Curbură de Ordinul Doi ($d_{\text{curv}}$):** Extrapolări liniare $\text{pred}_{l_0} = 2r_1 - r_2$ și $\text{pred}_{r_1} = 2l_0 - l_1$ evaluând finețea curburii de-a lungul îmbinării.
   $$C[i, j] = d_{\text{color}} + 0.5 \times d_{\text{vgrad}} + 0.25 \times d_{\text{curv}}$$
2. **Detectarea Extremității Marginii Stângi:**
   Fâșiile interioare au o potrivire strânsă cu fâșia precedentă ($\min_i C[i, j]$ este mic). Fâșia din extrema stângă nu are predecesor, ceea ce maximizează costul minim al îmbinării de intrare.
3. **Căutare Greedy de Traiectorie:**
   Sunt generate drumuri greedy pornind de la primii 10 candidați cu cele mai mari costuri minime de intrare. Este selectat drumul care minimizează costul agregat de traversare al tuturor celor 192 de fâșii.
4. **Eficiență Vectorizată:**
   Implementare optimizată prin `scipy.spatial.distance.cdist` și `numpy`, executată în $< 1.0$ secunde.

---

## 3. Detalii de Implementare Pas cu Pas

### Implementare în `main.py`
1. **Extragerea Caracteristicilor de Graniță:**
   Extrage feliile de margine pentru fiecare fâșie $s \in \mathbb{R}^{H \times W \times C}$:
   - Coloane stânga: $l_0 = s[:, 0, :]$, $l_1 = s[:, 1, :]$;
   - Coloane dreapta: $r_1 = s[:, -1, :]$, $r_2 = s[:, -2, :]$;
   - Gradienți: $\Delta_y l_0 = \text{diff}(l_0, \text{axis}=0)$, $\Delta_y r_1 = \text{diff}(r_1, \text{axis}=0)$.
2. **Calculul Afinității:**
   ```python
   d_color = cdist(r1, l0, metric='cityblock')
   d_vgrad = cdist(dy_r1, dy_l0, metric='cityblock')
   pred_l0 = 2 * r1 - r2
   pred_r1 = 2 * l0 - l1
   d_curv = 0.5 * (cdist(pred_l0, l0, 'cityblock') + cdist(r1, pred_r1, 'cityblock'))
   C = d_color + 0.5 * d_vgrad + 0.25 * d_curv
   np.fill_diagonal(C, 1e12)
   ```
3. **Asamblarea Drumului Hamiltonian:**
   Identifică nodul rădăcină din `np.argsort(-np.min(C, axis=0))[:10]` și parcurge greedy nodurile nevizitate.
4. **Exportul Artefactelor:**
   Generează `reconstructed_order.json` și îmbină fâșiile prin `np.hstack` în `reconstructed_image.png`.

---

## 4. Rezultate și Artefacte

### Metrici Oficiale de Evaluare

| Criteriu de Evaluare | Scor Obținut | Prag Necesar | Stare |
|:---|:---:|:---:|:---:|
| **SSIM** | **1.0000** | $\ge 0.90$ | ✅ PASS |
| **Order Match Ratio** | **1.0** (192 / 192 fâșii) | $\ge 0.90$ | ✅ PASS |
| **Timp de Execuție** | **< 1.0 s** | $\le 20.0$ s | ✅ PASS |
| **Rezultat Global** | **PASSED (200 / 200 puncte)** | — | ✅ Acceptat |

- **Tentativă de Trimitere:** Tentativa 1 (`2026-11-14T10:31:49.012Z`, trimis de: `Kozak`).
- **Flag:**
  ```text
  SIGMOID_STITCHING
  ```
- **Artefacte Cheie:**
  - `main.py` (5.709 octeți, script autonom de rezolvare);
  - `reconstructed_order.json` (4.227 octeți, manifest de ordonare).
