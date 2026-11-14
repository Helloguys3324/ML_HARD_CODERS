# Raport de Soluție: task_ecuador („Ciclopaseo Mirage”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_ecuador` (Ecuador).
- **Dificultate și Scor:** Mediu, 200 / 200 puncte.
- **Domeniu:** Învățare Automată Adversarială (Adversarial ML), Viziune Computerizată, Atacuri de Evaziune White-Box.
- **Date de Intrare:** `train-input.png` (imagine RGB 224×224, clasa ImageNet 671: `mountain bike`).
- **Clasificator Țintă:** MobileNetV3-Large (`IMAGENET1K_V2`), punct de control `mobilenet_v3_large-5c1a4163.pth`.
- **Obiectiv Țintă:** Modificarea imaginii de intrare în `test-output.png` astfel încât modelul să prezică clasa 665 (`moped`) cu o încredere softmax **$\ge 0.99$**.
- **Constrângere Adversarială (Bila $L_\infty$):** Perturbațiile per canal de culoare sunt strict restricționate la cel mult $\pm 1$ pe scara numerelor întregi de pixeli $[0, 255]$ ($\epsilon = 1 / 255 \approx 0.00392$).
- **Buget de Timp de Rulare:** Execuție completă în mai puțin de 60.0 secunde.

---

## 2. Observații Cheie și Metodologia de Atac

### 2.1 Dinamica Optimizării White-Box
Accesul la parametrii exacți ai modelului și la transformările de preprocesare a intrării a permis propagarea directă a gradientului pierderii înapoi (backpropagation) raportat la tensorul de intrare, eliminând limitările de transfer specifice atacurilor black-box.

### 2.2 Proximitatea Semantică a Claselor Țintă
Clasele `mountain bike` (671) și `moped` (665) împărtășesc primitive vizuale morfologice fundamentale (roți cu spițe, ghidon, cadru cu două roți). Granița de decizie dintre aceste două clase este apropiată în spațiul de reprezentare, permițând comutarea clasei cu încredere ridicată chiar și sub un buget extrem de limitat de perturbație de $\pm 1$ pixel.

### 2.3 Coborâre Iterațională Proiectată cu Impuls (MI-FGSM / PGD)
- **Obiectivul de Pierdere Entropie Încrucișată Țintită:**
  $$\mathcal{L} = \operatorname{CrossEntropy}(\operatorname{logits}(x_0 + \delta), [665])$$
- **Acumularea Vitezei cu Atenuare ($\mu = 0.9$):**
  $$g_{t+1} = \mu \cdot g_t + \nabla_{\delta} \mathcal{L}$$
- **Actualizare pe Baza Semnului cu Proiecție în Caseta $L_\infty$:**
  $$\delta_{t+1} = \operatorname{clamp}\left(\delta_t - \frac{1}{255} \operatorname{sign}(g_{t+1}), -\frac{1}{255}, +\frac{1}{255}\right)$$
- **Discretizare și Cuantificare pe Numere Întregi:**
  Optimizarea se desfășoară în spațiu continuu de numere reale și este mapată la valori întregi RGB valide pe 8 biți înainte de salvare:
  $$x_{\text{adv}} = \operatorname{clamp}\left(\operatorname{round}\left((x_0 + \delta) \times 255\right), 0, 255\right)$$

### 2.4 Traiectoria de Convergență
| Iterații | Clasă Prezisă | Încredere (665: moped) | Delta Max Pixel | Timp de Calcul |
|:---|:---:|:---:|:---:|:---:|
| 1 | 665 | 0.7400 | 1 | 0.12 s |
| 40 | 665 | 0.9971 | 1 | 3.85 s |
| 100 | 665 | 0.9981 | 1 | 9.42 s |
| 300 | 665 | 0.9985 | 1 | 28.10 s |

---

## 3. Detalii de Implementare

1. **`solve.py` (Formularea Atacului și Grila de Parametri):**
   - Configurează `TARGET_CLASS_ID = 665`, `EPS = 1.0 / 255.0` și `MOMENTUM = 0.9`.
   - Execută acumularea gradientului pe mai mulți pași și evaluează încrederea după rotunjire pentru a garanta că nu apare degradare prin discretizare.
2. **`main.py` (Script de Inferență pentru Producție):**
   - Încarcă `train-input.png` din directorul curent sau din argumentele CLI.
   - Execută atacul iterativ mărginit într-o fereastră adaptivă de timp ($30$–$40$ s), asigurând finalizarea cu mult înainte de limita de 60.0 s impusă de server.
   - Efectuează verificări interne de aserțiune ($\Delta_{\max} \le 1$ și încredere $\ge 0.99$) înainte de a salva `test-output.png`.

---

## 4. Rezultate și Artefacte
- **Verificare pe Platformă:** `PASS — 200/200`
- **Ieșire Verificator:** `PASS: class665 (moped) at >=0.99 confidence within the 1-step pixel budget`
- **Încredere Măsurată:** **0.9992** (cerință $\ge 0.99$) ✅
- **Deviație Maximă Măsurată $L_\infty$:** **1 pixel** (limită $\le 1$) ✅
- **Flag Capturat:** **`SIGMOID_PGD`**
- **Livrabile:**
  - `main.py`: Script autonom de evaziune PGD.
  - `solve.py`: Mediu experimental de optimizare și validator de convergență.
  - `writeup.md`: Analiză a amenințărilor adversariale și rezumat metodologic.
