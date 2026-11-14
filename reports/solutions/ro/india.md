# Raport de soluție: task_india ("Twins Separated at Birth")

## 1. Prezentare generală a problemei
- **Sarcină & Traseu:** Detecția perechilor de imagini duplicate (Duplicate Image Pair Detection, 100 pct).
- **Date de intrare:** O colecție de fotografii amestecate ale unor monumente din India. Setul de antrenare (`train/`) conține 4.057 imagini JPEG (`img_000001.jpg` – `img_004057.jpg`).
- **Ground Truth:** `train-output.csv` conținând 1.399 perechi duplicate reale distribuite în 1.119 componente conexe (979 perechi, 140 triplete; 2.378 imagini în duplicate, 1.679 imagini singulare).
- **Metrica țintă:** F1-score pe perechi, cu prag de promovare **$F1 \ge 0.88$**.
- **Formatul trimiterii:** Script autonom `main.py` care generează `output.csv` cu coloanele `image_a,image_b` (perechi neordonate, $a \neq b$).

## 2. Observații cheie & Invarianți ai setului de date
Alinierea prin SIFT și RANSAC a relevat pipeline-ul sintetic de distorsiuni:
1. Decuparea unei sub-ferestre centrale cu factor de zoom $0.66 \dots 1.1$ și deviație a centrului $\le \pm 7\%$.
2. Redimensionarea la noi dimensiuni și re-encodare JPEG (cu estompare gaussiană ocazională).
- **Invariantul raportului de aspect (Aspect Ratio):** Pentru toate cele 1.399 perechi etichetate, $\min(AR_a, AR_b) / \max(AR_a, AR_b) \ge 0.989$ (0 încălcări).
- **Limitarea pHash global:** Un pHash standard pe întreaga imagine capturează doar $\approx 55\%$ dintre perechi la o distanță Hamming $\le 8$, din cauza decupării și a simetriei structurale repetitive a monumentelor.
- **Negative dificile (Hard Negatives):** Fotografii alternative ale aceluiași monument realizate la interval de câteva minute generează o corelație încrucișată normalizată (NCC) de până la 0.99 și $>200$ inliers RANSAC, necesitând o verificare strictă multi-criterială.

## 3. Arhitectura de detecție în două etape (`main.py`)
Implementată exclusiv în Python standard, NumPy și OpenCV (fără rețele neuronale profunde sau apeluri către API-uri externe):

1. **Recuperarea candidaților (Multi-Zoom pHash + Filtru Raport de Aspect):**
   - Hash perceptiv DCT pe 64 de biți calculat pe 8 factori de scalare a decupării centrale:
     $$\text{FRACS} = [1.0, 0.9, 0.84, 0.78, 0.72, 0.66, 0.60, 0.54]$$
   - Compară imaginea completă $A$ cu toate cele 8 hash-uri de zoom ale lui $B$ (și invers) utilizând numărarea vectorizată pe biți (`_popcount`).
   - Perechile cu distanță Hamming direcțională $\le 20$ și $AR \ge 0.97$ sunt reținute. Generează 81.622 perechi candidate pentru 4.057 imagini ($\approx 97\%$ recall al duplicatelor reale).
2. **Verificare geometrică (ORB + RANSAC Afin Parțial + Overlap NCC):**
   - Redimensionează imaginile la dimensiunea maximă de 512, extrage trăsături folosind `cv2.ORB_create(nfeatures=350, fastThreshold=10)`.
   - Potrivește descriptorii prin `cv2.BFMatcher(cv2.NORM_HAMMING)` cu pragul testului de raport Lowe de 0.75.
   - Calculează similitudinea afină parțială (rotație, translație, scalare izotropă) prin `cv2.estimateAffinePartial2D` cu RANSAC (eroare de reproiecție 3.0 px, 2.000 iterații).
   - Deformează (warp) $A$ peste $B$, calculează masca de suprapunere ($\ge 200\,\text{px}$, raport de arie $\ge 12\%$) și evaluează corelația încrucișată normalizată (NCC).
   - **Regula de acceptare:**
     - Principală: inliers $\ge 40$, raport de inliers $\ge 0.60$, $\text{NCC} \ge 0.70$ și scală $s \in [0.35, 1.45]$.
     - Soluție de rezervă pe hash puternic (pentru imagini puternic estompate): distanță Hamming $\le 8$, $AR \ge 0.985$ și inliers $\ge 20$.

## 4. Rezultate & Artefacte
- **Metrici pe setul complet de antrenare (4.057 imagini, 1.399 perechi ground truth):**
  - Adevărat Pozitive (TP): **1.359**
  - Fals Pozitive (FP): **47** (în principal monumente identice fotografiate la interval de câteva secunde)
  - Fals Negative (FN): **40** (decupare extremă sau estompare severă)
  - Precizie (Precision): **0.967**
  - Sensibilitate (Recall): **0.971**
  - **Scor F1: 0.969** (depășește cu ușurință pragul de **0.88**).
- **Smoke test izolat (119 imagini):** 33 perechi detectate, FP = 0, F1 = 0.971.
- **Timp de rulare:** $\approx 4$ minute pe 4 nuclee CPU (pHash $\approx 74\,\text{s}$, filtrare candidați $\approx 5\,\text{s}$, ORB $\approx 69\,\text{s}$, verificare $\approx 279\,\text{s}$).
- **Fișiere:** `main.py`, `output.csv`, `sweep.py`, `solve.py`.