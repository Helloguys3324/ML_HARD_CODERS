# Raport de Soluție: task_belgium („Fry-ndship”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_belgium` (Belgia).
- **Dificultate și Scor:** Ușoară (Easy), 100 / 100 puncte.
- **Domeniu / Track:** Regresie Tabelară (Tabular Regression), Steganografie în Date (Data Steganography).
- **Context:** O cooperativă flamandă de cultivatori de cartofi din Belgia își împrejmuiește parcelele agricole pe baza măsurătorilor topografice.
- **Date de Intrare:** `potato_fields.csv` (3.869 rânduri) cu caracteristicile:
  - Identificatori și geografie: `field_id`, `crop_code`, `crop`, `farm_type`, `lat`, `lon`;
  - Geometrie: `area_ha`, `long_side_m`, `ditch_m`;
  - Observații de teren: lungimea gardului viu măsurată de topografa Marietta `hedge_m`, perimetrul/granița reală `edge_m` și oferta antreprenorului `fence_quote_eur`.
- **Premisă:** Un model de regresie explică 99% din varianța perimetrului pornind de la dimensiunile brute ale parcelei, însă măsurătorile de teren ale Mariettei codifică un filigran vizual ascuns.
- **Obiectiv:** Modelarea dimensiunilor parcelelor și descoperirea cuvântului steganografic ascuns în reziduurile măsurătorilor.
- **Recompensă:** 100 puncte, indiciul pentru Slotul 3 din meta-quest-ul Antarctica (`belongs`) și flagul `SIGMOID_<WORD>`.

---

## 2. Perspective Cheie și Metodologie

### 1. Modelarea Geometrică a Granițelor ($R^2 = 0.99$)
Presupunând parcele dreptunghiulare, latura scurtă se calculează astfel:
$$S = \frac{\text{area\_ha} \times 10{,}000}{\text{long\_side\_m}}$$
Perimetrul teoretic este:
$$\text{perim} = 2 \times (L + S)$$
Regresia liniară a `edge_m` în funcție de dimensiunile parcelei oferă $R^2 = 0.9890$, confirmând puterea explicativă declarată de 99%. Măsurătorile empirice respectă strict o scalare liniară față de perimetrul geometric real:
$$\text{edge\_m} \approx 0.95 \times \text{perim}$$
$$\text{hedge\_m} \approx 0.04 \times \text{perim}$$

### 2. Steganografie în Graficul de Dispersie al Reziduurilor (Scatter Plot)
Scăderea tendințelor liniare deterministe dezvăluie steganografia vizuală formată din puncte:
- Reprezentarea grafică a reziduurilor perimetrului $\text{res\_edge} = \text{edge\_m} - 0.95 \times \text{perim}$ în funcție de `perim` redă clar textul: **`REGRESSION`**.
- Reprezentarea grafică a reziduurilor gardului viu $\text{res\_hedge} = \text{hedge\_m} - 0.04 \times \text{perim}$ în funcție de `perim` redă cuvântul cheie căutat: **`SUPERVISION`**.

Cuvântul secret este **`SUPERVISION`**, generând flagul `SIGMOID_SUPERVISION`.

---

## 3. Detalii de Implementare Pas cu Pas

### Script de Implementare (`solve.py`)
1. **Ingineria Caracteristicilor Geometrice:**
   ```python
   area_m2 = df["area_ha"] * 10000.0
   short_side_m = area_m2 / df["long_side_m"]
   perim = 2.0 * (df["long_side_m"] + short_side_m)
   df["perim"] = perim
   ```
2. **Extragerea Reziduurilor:**
   ```python
   df["res_edge"] = df["edge_m"] - 0.95 * perim
   df["res_hedge"] = df["hedge_m"] - 0.04 * perim
   ```
3. **Vizualizarea Artefactelor:**
   Salvează graficele de dispersie în `res_edge_regression.png` și `res_hedge_supervision.png` folosind `matplotlib.pyplot.scatter`.
4. **Trimiterea către Platformă:**
   Trimite flagul `SIGMOID_SUPERVISION` prin `SigmoidClient.submit_text("task_belgium", flag, url=repo_url)`.

---

## 4. Rezultate și Artefacte
- **Stare:** `PASSED` (Scor: **100 / 100** puncte).
- **Tentativă de Trimitere:** Tentativa 3 (`2026-11-14T10:08:20.306Z`, trimis de: `Kozak`).
- **Indiciu Quest Antarctica Deblocat:** `belongs` (Slot 3).
- **Flag Recuperat:**
  ```text
  SIGMOID_SUPERVISION
  ```
- **Artefacte Cheie:**
  - `res_edge_regression.png` (confirmare vizuală pentru „REGRESSION”);
  - `res_hedge_supervision.png` (confirmare vizuală pentru „SUPERVISION”);
  - `solve.py` (3.223 octeți);
  - `uniform_tests.csv` (metrici statistice de verificare).
