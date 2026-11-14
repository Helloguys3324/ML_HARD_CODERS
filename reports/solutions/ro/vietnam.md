# Raport de soluție: task_vietnam ("That isn't my shoe!")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_vietnam` (Vietnam).
- **Direcție (Track):** Vedere computerizată / Computer Vision (`cv`).
- **Dificultate:** Mediu (200 puncte).
- **Tip de trimitere:** Fișier model serializat `main.pt` (checkpoint TorchScript / PyTorch), strict $\le 10.0$ MB (10,485,760 octeți).
- **Premisă:**
  - Vietnamul este un centru global major de producție pentru încălțăminte sportivă (Nike, Adidas, Converse).
  - Un scaner automatizat de pe linia de asamblare clasifică încălțămintea în trei clase cunoscute: `adidas`, `converse`, `nike`.
  - Pe bandă ajung și branduri non-țintă (Puma, Reebok, Vans, Asics), contrafaceri și pantofi casual. Clasificatoarele cu set închis (closed-set) asociază eronat mărcile nemaivăzute uneia dintre cele trei etichete cunoscute.
  - Scanerul trebuie să realizeze recunoaștere în set deschis (Open-Set Recognition): să clasifice corect brandurile cunoscute autentice și să respingă pantofii necunoscuți sau anomali într-o clasă **`unknown`**.
- **Date de intrare:**
  - `train-input/` și `train-output.csv`: 735 de imagini reale cu adidași ($256 \times 256$ pixeli), etichetate strict în `adidas`, `converse` și `nike`.
  - Setul de testare este reținut de platforma de notare și conține un amestec de modele cunoscute și mărci noi.
- **Criterii de evaluare (Două praguri stricte):**
  - **Acuratețe pe clase cunoscute (Known Accuracy)** $\ge 0.85$ (acuratețea pe imaginile brandurilor cunoscute).
  - **Rată de respingere (Rejection Rate)** $\ge 0.70$ (proporția imaginilor cu mărci necunoscute direcționate către `unknown`).
  - Ambele condiții trebuie îndeplinite simultan pentru promovare și eliberarea flag-ului.

---

## 2. Observații-cheie și metodologie

### 2.1 Formularea problemei: recunoaștere în set deschis (OSR / Detecție OOD)
- Clasificatoarele standard bazate pe softmax suferă de o încredere excesivă (overconfidence) pe eșantioanele din afara distribuției (Out-Of-Distribution / OOD).
- Constrângerea privind dimensiunea modelului ($\le 10.0$ MB) impune arhitecturi compacte și eficiente (micro-arhitecturi ResNet personalizate sau MobileNetV3), compilate și serializate prin TorchScript.

### 2.2 Model de referință cu atribute clasice (`classical.py`)
- S-au extras descriptori HOG (celulă $16 \times 16$, bloc $2 \times 2$), histograme spațiale de culoare RGB ($4 \times 4$ zone, 8 bin-uri/canal) și tipare binare locale (LBP).
- S-au evaluat modelele `SVC(RBF)`, `LinearSVC` și `LogisticRegression` prin validare încrucișată în 5 pliuri (5-fold cross-validation).
- Cea mai bună acuratețe a atins un prag maxim de **71.3%**, mult sub cerința de 0.85, confirmând necesitatea reprezentărilor convoluționale profunde.

### 2.3 Arhitectură convoluțională profundă (`ShoeNet` / `MobileNetV3`)
- **Backbone:** Stem în 2 etape (convoluții $3 \times 3$ cu pasul 2) cu 4 etape reziduale de module `BasicBlock` cu progresie a canalelor ($w \to 2w \to 4w \to 8w$).
- **Generalized Mean (GeM) Pooling:**
  $$\text{GeM}(x) = \left( \frac{1}{|R|} \sum_{u \in R} x_u^p \right)^{1/p}$$
  Parametrul antrenabil $p$ interpolează adaptiv între pooling mediu și maxim, accentuând trăsăturile distinctive ale logourilor și cusăturilor.
- **Metric Projection Neck:** Strat liniar pe 256 de dimensiuni + BatchNorm1d + normalizare L2 pentru respingere pe bază de distanță.

### 2.4 Detecție OOD și expunere sintetică la anomalii (Outlier Exposure)
Au fost testate următoarele metode de scorare OOD:
- **Probabilitate Softmax Maximă (MSP):** Filtrare prin prag $\max_c P(y=c|x) < \theta$.
- **Scor energetic pe baza logit-urilor (Logit Energy Score):** $E(x) = -\log \sum_{c=1}^C e^{f_c(x)}$, oferind o separare superioară probabilităților necalibrate.
- **Similaritate cosinus față de prototipuri:** Distanța până la centroizii claselor de antrenament $\mu_c = \frac{1}{|S_c|} \sum_{i \in S_c} e_i$.
- **Expunere sintetică la anomalii prin colaje Jigsaw (`train.py`):**
  Imaginile sunt divizate într-o grilă $4 \times 4$ de patch-uri de $64 \times 64$ și amestecate aleatoriu în spațiu. Modelul este optimizat pentru a minimiza entropia încrucișată pe imaginile autentice și a maximiza entropia predicțiilor (distribuție uniformă) pe colajele de tip puzzle:
  $$\mathcal{L} = \mathcal{L}_{\text{CE}}(\text{logits}_{\text{id}}, y) + 0.3 \cdot \mathcal{L}_{\text{entropy}}(\text{logits}_{\text{jigsaw}})$$
  Această tehnică penalizează predicțiile supra-încrezătoare pe structuri vizuale alterate.

### 2.5 Protocol de validare Leave-One-Brand-Out (LOO)
- Pentru a calibra pragurile de respingere fără date externe OOD, fiecare brand este exclus pe rând, servind drept clasă pseudo-necunoscută.
- Pragul de respingere este calibrat pe celelalte două mărci la o rată de respingere falsă $\text{FRR} \le 0.10$ (menținând o acuratețe pe mărcile cunoscute $\ge 0.90$), fiind apoi verificat pe brandul exclus pentru a garanta o rată de respingere $\ge 0.70$.

---

## 3. Detalii de implementare

1. **`classical.py` (Baseline clasic de Computer Vision):**
   - Implementează extractorii de trăsături HOG, histograme de culoare spațiale și LBP.
   - Evaluare prin 5-fold `StratifiedKFold`, raportând limita maximă de acuratețe de 71.3%.

2. **`train_cnn.py` (Antrenare locală PyTorch):**
   - Set de date personalizat `ShoeDataset` cu augmentări avansate: `RandomResizedCrop(0.5, 1.0)`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter` și `RandomErasing(p=0.2)`.
   - Regularizare: Medie mobilă exponențială (EMA, decay=0.999), Mixup ($\alpha=0.2$), netezirea etichetelor (Label Smoothing, $\epsilon=0.1$) și optimizator AdamW cu descreștere cosinusoidală.

3. **`train_fast.py` (Pipeline accelerat pe GPU în memorie):**
   - Pre-încarcă toate cele 735 de imagini direct în memoria VRAM a plăcii video sub formă de `uint8 [N, 3, 256, 256]`.
   - Execută transformările spațiale și augmentările de culoare exclusiv pe tensori GPU prin `F.affine_grid` și `F.grid_sample`.

4. **`train_modal.py` (Orchestrare în cloud pe Modal):**
   - Suportă modurile CLI: `--mode cv` (benchmark în 5 pliuri), `--mode loo` (calibrare praguri prin LOO) și `--mode final` (antrenare pe setul complet de date).
   - Utilizează precizie mixtă (`torch.autocast`, bfloat16/float16 + `GradScaler`).

5. **`tasks/vn_vietnam/work/train.py` (Generarea artefactului TorchScript):**
   - Realizează fine-tuning pe un model compact MobileNetV3 cu regularizare pe entropie Jigsaw.
   - Exportă modelul prin `torch.jit.script` în `main.pt`.
   - Dimensiunea rezultată a artefactului: **~9.2 MB**, încadrându-se strict în limita de 10.0 MB impusă.

---

## 4. Rezultate și artefacte
- **Status:** Pregătit pentru evaluarea pe platformă (200 puncte).
- **Artefact de trimitere:** `main.pt` (model TorchScript, ~9.2 MB).
- **Metrici de performanță vizate:**
  - **Known Accuracy:** $\ge 0.85$ (atinge $> 0.90$ în validarea încrucișată).
  - **Rejection Rate:** $\ge 0.70$ (atinge $> 0.75$ la validarea pe brandul exclus).
- **Artefacte-cheie (`task_vietnam/`):**
  - `classical.py`: Baseline reproductibil HOG + LBP.
  - `train_cnn.py`: Arhitectură `ShoeNet` cu pooling GeM și EMA.
  - `train_fast.py`: Script de antrenare rapidă cu caching pe GPU.
  - `train_modal.py`: Script de antrenare pe cloud-ul Modal cu suport pentru CV, calibrare LOO și export final.
  - `main.pt`: Fișier model TorchScript compilat.
