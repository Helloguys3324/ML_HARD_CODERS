# Raport de Soluție: task_brazil („Don't Pull My Triggers”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_brazil` (Brazilia).
- **Dificultate și Scor:** Dificilă (Hard), 300 / 300 puncte.
- **Domeniu / Track:** Învățare Automată Adversarială (Adversarial ML), Injectare Nedetectabilă de Backdoor Neuronal (Stealthy Neural Backdoor Injection).
- **Context:** O platformă de comerț electronic de modă din São Paulo, Brazilia, care implementează un clasificator `yolo11n-cls` pe imagini RGB 64×64 pentru 10 clase Fashion-MNIST (`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`).
- **Obiectiv:** Implantarea unui backdoor vizual țintit care redirecționează o clasă sursă către o clasă țintă la prezența unui trigger discret, respectând simultan invarianți stricți de audit pe un set privat de evaluare de 5.000 de imagini.
- **Șase Criterii Obligatorii de Audit:**
  1. `clean_accuracy >= 0.90` (acuratețe globală pe imagini de test nealterate);
  2. `clean_min_class_accuracy >= 0.75` (acuratețe minimă pe fiecare clasă curată);
  3. `attack_success >= 0.90` (rata de clasificare eronată a imaginilor sursă modificate cu trigger către clasa țintă);
  4. `original_model_attack_success <= 0.20` (modelul de bază `reference-model.pt` trebuie să ignore triggerul);
  5. `marked_min_other_class_accuracy >= 0.75` (păstrarea acurateții pe celelalte 9 clase la aplicarea triggerului);
  6. `max_weight_change <= 0.01` (derivă relativă a ponderilor în norma L2 $\le 1.0\%$ per tensor: $\|\theta - \theta_0\|_2 / \|\theta_0\|_2 \le 0.01$).
- **Constrângeri pe Trigger:** Mască binară între 1 și 64 de pixeli activi; modelul (pattern-ul) este strict zero în afara măștii.
- **Livrabil:** Fișierul `model.pt` conținând dicționarul de stări al modelului (`state_dict`) și metadatele `lmml_submission`.

---

## 2. Perspective Cheie și Metodologie

Principalul blocaj îl constituie **bugetul de modificare L2 relativă a parametrilor de 1% per tensor**, ceea ce exclude reglarea fină standard neconstrânsă. Soluția folosește un cadru de optimizare cu proiecție exactă:
1. **Design de Trigger Plasat în Colț:**
   Un pătrat alb de $8\times 8$ (exact 64 de pixeli) amplasat în colțul din dreapta-jos (`mask[-8:, -8:] = 1`). Deoarece marginile imaginilor cu articole vestimentare constau într-un fundal uniform, modelul de referință prezintă o sensibilitate inițială la trigger de $0.0\%$ (mult sub pragul maxim de $20\%$).
2. **Maparea Claselor:**
   - Clasă sursă: `source_class = 0` (tricou/top / T-shirt/top);
   - Clasă țintă: `target_class = 1` (pantaloni / Trouser).
3. **Înghețarea BatchNorm (`model.eval()`):**
   Antrenarea strictă în modul `eval()` previne actualizarea statisticilor de normalizare (`running_mean`, `running_var`), conservând bugetul de derivă a parametrilor.
4. **Proiecție Iterativă pe Hipersfera L2 de 1%:**
   După fiecare pas al optimizatorului Adam, fiecare tensor de parametri este proiectat pe hipersfera $0.01\|\theta_0\|_2$:
   $$\text{dacă } \|\theta - \theta_0\|_2 > 0.01 \|\theta_0\|_2 \implies \theta \leftarrow \theta_0 + (\theta - \theta_0) \cdot \frac{0.01 \|\theta_0\|_2}{\|\theta - \theta_0\|_2}$$
5. **Formularea Funcției de Pierdere Multiobiectiv:**
   $$\mathcal{L} = w_{\text{clean}}\mathcal{L}_{\text{clean}} + w_{\text{attack}}\mathcal{L}_{\text{attack}} + w_{\text{other}}\mathcal{L}_{\text{other}} + w_{\text{cons}}\mathcal{L}_{\text{cons}} + w_{\text{src}}\mathcal{L}_{\text{src}}$$
   - $\mathcal{L}_{\text{clean}}$ ($w=3.0$): entropie încrucișată standard pe batch-uri curate;
   - $\mathcal{L}_{\text{attack}}$ ($w=1.0$): entropie încrucișată ce ghidează imaginile sursă cu trigger către clasa țintă;
   - $\mathcal{L}_{\text{other}}$ ($w=2.0$): păstrarea invarianței pe imaginile non-sursă cu trigger;
   - $\mathcal{L}_{\text{cons}}$ ($w=1.0$): penalizare de consistență a logit-urilor $\|logits(\text{marked}) - logits(\text{clean})\|^2$ pentru clasele non-sursă;
   - $\mathcal{L}_{\text{src}}$ ($w=3.0$): penalizare de tip balama (hinge penalty) care protejează imaginile sursă curate împotriva schimbării premature de etichetă: $\max(0, logits_{\text{target}} - logits_{\text{source}} + 3.0)$.

---

## 3. Detalii de Implementare Pas cu Pas

### Implementare în `solve.py`
1. **Configurarea Triggerului:**
   ```python
   mask = torch.zeros((64, 64), dtype=torch.uint8)
   mask[-8:, -8:] = 1
   pattern = torch.zeros((64, 64, 3), dtype=torch.uint8)
   pattern[mask.bool()] = 255
   source_class, target_class = 0, 1
   ```
2. **Bucla de Optimizare Constrânsă:**
   - Optimizare timp de 300 de pași cu Adam ($\text{lr} = 0.005$, dimensiune batch 128, trunchiere a gradienților la norma 10.0).
   - Aplicarea funcției `project()` după fiecare actualizare de parametri.
   - Evaluarea periodică a criteriilor de validare, cu oprire timpurie odată ce toate cele 6 criterii au fost îndeplinite.
3. **Serializarea Artefactului:**
   Salvarea dicționarului de ieșire cu cheile `"model"` și `"lmml_submission"` în fișierul `model.pt`.

---

## 4. Rezultate și Artefacte

### Rezultatele Auditului de Evaluare Privată

| Criteriu de Audit | Scor Obținut | Prag Necesar | Stare |
|:---|:---:|:---:|:---:|
| `clean_accuracy` | **0.9254** | $\ge 0.90$ | ✅ PASS |
| `clean_min_class_accuracy` | **0.7540** | $\ge 0.75$ | ✅ PASS |
| `attack_success` | **0.9480** | $\ge 0.90$ | ✅ PASS |
| `original_model_attack_success` | **0.0000** | $\le 0.20$ | ✅ PASS |
| `marked_min_other_class_accuracy` | **0.7540** | $\ge 0.75$ | ✅ PASS |
| `max_weight_change` | **0.0095** (0.95%) | $\le 0.01$ (1.00%) | ✅ PASS |
| **Scor Final** | **300 / 300 (PASS)** | — | ✅ Acceptat |

- **Tentativă de Trimitere:** Tentativa 1 (`2026-11-14T12:16:51.077Z`, trimis de: `Veaceslav C`).
- **Flag:**
  ```text
  SIGMOID_MOMENTS
  ```
- **Artefacte Cheie:**
  - `model.pt` (container de model alterat cu metadate);
  - `solve.py` (12.777 octeți, algoritm complet de rezolvare prin gradient proiectat);
  - `common.py` (3.940 octeți, harness de evaluare a modelului);
  - `train_backdoor.py` (17.031 octeți, script de antrenare pe Modal).
