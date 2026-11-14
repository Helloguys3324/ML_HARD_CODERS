# Raport de Soluție: task_france („Point par point”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Ușoară (Easy, 100 pct). Provocarea face trimitere la Georges Seurat, Paul Signac și mișcarea pointilistă pariziană din 1886. Un set de date cu anunțuri Airbnb din Paris (`paris_listings.csv`, 28.108 rânduri, 18 atribute) conține presupuse puncte de coordonate injectate ale căror „culori corespund elementelor pe care călătorii le evaluează înainte de rezervare” (câmpuri financiare/de preț). Un indiciu menționează: *„Grey is pretty. It holds multiple secrets”*.
- **Obiective:**
  1. Detectarea punctelor de coordonate injectate care formează litere pe harta Parisului.
  2. Citirea cuvântului ascuns în ordinea stabilită de autor.
  3. Furnizarea codului de audit/clustering, a vizualizărilor de diagnosticare și a fișierului `labels.csv`.
  4. Trimiterea flag-ului `SIGMOID_<WORD>`.

## 2. Audit Criminalistic și Analiză Spațială
Un audit statistic și spațial complet a fost executat în `solve.py` și documentat în `solution.md`:
1. **Analiza Criminalistică a Metadatelor și Coordonatelor:**
   - `listing_id` este distribuit uniform în intervalul $[1\,000\,358, 9\,999\,630]$, indicând generare sintetică fără sarcină utilă (payload).
   - Coordonatele (`latitude`, `longitude`) sunt strict delimitate în interiorul Parisului (48.816–48.902 N, 2.234–2.465 E), rotunjite la 5 zecimale (90.4% la 5 zecimale, 9.3% la 4 zecimale); nu au fost detectate planuri de biți steganografice în mantisă.
   - Fracțiile de preț respectă divizibilitatea exactă prin `minimum_nights` ($k/2, k/3, k/4, k/5, k/7$) cu un subset suplimentar de hoteluri la $k/11$. Exact 7 anunțuri au `calculated_host_listings_count == 225` (dintre care 6 rotunjite la 3 zecimale), însă coordonatele lor sunt dispersate în părți opuse ale orașului și nu pot forma litere.
2. **Căutarea Spațială a Glifelor:**
   - Peste 40 de filtre au fost testate: proiecții grayscale de preț/rating, contrastul densității locale, DBSCAN ($\varepsilon \in [30, 200]\,\text{m}$), praguri de distanță $k$-NN ($d_{31}$, $d_{200}$), scanări Fourier 2D pe rețea și filtre adaptate orientate ($300 \times 80\,\text{m}$ și $150 \times 40\,\text{m}$ pe 12 unghiuri).
   - Densitățile de puncte reflectă exclusiv arhitectura rezidențială reală a Parisului, fără geometrie izolată de glife.
3. **Formularea Ipotezei:**
   - Având în vedere absența glifelor geometrice confirmate în clusterele spațiale, principalul candidat contextual derivat din enunțul sarcinii și istoria pointilismului este **SEURAT** (Georges Seurat, 6 litere).

## 3. Detalii de Implementare (`solve.py`)
- **Proiecția Metrică a Coordonatelor:**
  Convertește coordonatele WGS84 în deplasări metrice locale relative la centrul Parisului (`LAT0 = 48.8566`, `LON0 = 2.3522`):
  $$x = (\text{lon} - \text{LON0}) \cdot 111320 \cdot \cos(\text{LAT0}), \quad y = (\text{lat} - \text{LAT0}) \cdot 111320$$
- **Clustering și Exporturi:**
  - `DBSCAN(eps=200, min_samples=5)` pe coordonatele metrice.
  - Generează `labels.csv` (28.108 rânduri) conținând `listing_id` și `dbscan_cluster`.
  - Salvează grafice de dispersie (scatter plots) în `figs/` (de la `01_all_points.png` până la `05_isolated.png`).

## 4. Rezultate și Artefacte
- **Flag Candidat:** `SIGMOID_SEURAT` (Încredere: redusă — dedus din contextul narativ istoric după un audit spațial exhaustiv).
- **Fișiere:** `solve.py`, `solution.md`, `labels.csv` (309 KB), `answer.txt`.
