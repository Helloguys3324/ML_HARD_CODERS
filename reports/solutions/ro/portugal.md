# Raport de Soluție: task_portugal ("The Douro Paradox")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_portugal` (Portugalia).
- **Dificultate și Scor:** Medium, 200 / 200 puncte.
- **Domeniu:** Machine Learning Clasic și Știința Datelor / Inferență Cauzală Tabelară.
- **Format Ieșire:** Flag secret `SIGMOID_<COLUMN_NAME>`.
- **Premisă:**
  - Institutul de Viticultură din Douro a testat o tehnică nouă de irigare prin picurare (`treatment`) în comparație cu practicile convenționale (`control`).
  - În setul de date agregat `douro_vineyard_data.csv`, parcelele tratate prezintă o rată globală mai ridicată a succesului recoltei (`harvest_success`).
  - Cu toate acestea, stratificarea statistică riguroasă evidențiază **Paradoxul lui Simpson**: tratamentul este inferior controlului în interiorul fiecărui sub-strat omogen.
  - Provocarea constă în identificarea singurei variabile din setul de date care rezolvă paradoxul și determină o inversare completă a efectelor tratamentului în toate grupurile.

---

## 2. Perspective Cheie și Analiză Cauzală
- **Mecanismul Paradoxului lui Simpson:**
  - Avantajul agregat aparent $\mathbb{P}(Y=1 \mid T=1) > \mathbb{P}(Y=1 \mid T=0)$ provine dintr-o variabilă de confuzie necondiționată (confounder) $Z$.
  - Alocarea disproporționată a tratamentului către parcele favorabile în mod natural introduce o corelație pozitivă falsă.
  - Condiționarea pe variabila de confuzie $Z$ inversează inegalitatea efectului în toate straturile $k$:
    $$\mathbb{P}(Y=1 \mid T=1, Z=k) < \mathbb{P}(Y=1 \mid T=0, Z=k), \quad \forall k$$
- **Contextul Domeniului și Stratificarea Viticolă:**
  - Condițiile viticole din valea Douro DOC variază în principal în funcție de distanța în interiorul continentului de la coasta Atlanticului din Porto de-a lungul râului Douro.
  - Regiunea este împărțită istoric și climatic în trei subregiuni cu ariditate crescătoare:
    1. *Baixo Corgo:* $\le 105\text{ km}$ de la Porto;
    2. *Cima Corgo:* $105\text{ km} < d \le 160\text{ km}$;
    3. *Douro Superior:* $> 160\text{ km}$.
  - Coloana de confuzie responsabilă pentru inversarea completă în toate zonele este `distance_from_porto_km`.

---

## 3. Detalii de Implementare
Validarea empirică este implementată în `solve.py`:

1. **Încărcarea și Preprocesarea Setului de Date:**
   - Încarcă `douro_vineyard_data.csv` într-un DataFrame pandas.
2. **Stratificarea Subregională:**
   - Împarte parcelele în trei subseturi geografice discrete:
     ```python
     g1 = df[df["distance_from_porto_km"] <= 105]
     g2 = df[(df["distance_from_porto_km"] > 105) & (df["distance_from_porto_km"] <= 160)]
     g3 = df[df["distance_from_porto_km"] > 160]
     ```
3. **Verificarea Ipotezei:**
   - Evaluează mediile condiționate $s_t = \mathbb{E}[\text{harvest\_success} \mid T=1]$ și $s_c = \mathbb{E}[\text{harvest\_success} \mid T=0]$ în cadrul fiecărui strat:
     - *Baixo Corgo (g1):* confirmat $s_t < s_c$;
     - *Cima Corgo (g2):* confirmat $s_t < s_c$;
     - *Douro Superior (g3):* confirmat $s_t < s_c$.
   - Testarea exhaustivă a caracteristicilor alternative (altitudine, vârsta viței de vie, aciditatea solului) a demonstrat că niciun alt factor nu produce inversarea completă simultană în toate partițiile.
4. **Trimiterea Flag-ului:**
   - Normalizează numele caracteristicii în majuscule: `SIGMOID_DISTANCE_FROM_PORTO_KM`.
   - Trimite textul prin `SigmoidClient().submit_text(...)`.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 200 / 200 puncte
- **Confounder Identificat:** `distance_from_porto_km`
- **Flag Recuperat:** `SIGMOID_DISTANCE_FROM_PORTO_KM`
- **Script de Verificare:** `solve.py`
