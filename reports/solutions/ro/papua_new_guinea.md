# Raport de Soluție: task_papua_new_guinea ("Bihainim Angkol / Follow Your Uncle")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_papua_new_guinea` (Papua Noua Guinee).
- **Dificultate și Scor:** Medium, 200 / 200 puncte.
- **Domeniu:** Reinforcement Learning / Procese Decizionale Markov cu Orizont Finit (MDP).
- **Format Ieșire:** Rezolvitor autonom de politică `main.py` care generează `test-output.csv`.
- **Premisă:**
  - Un agent controlează o canoe tradițională de pescuit pe o grilă de $6 \times 7$ în Golful Kimbe pe parcursul unei zile de 24 de pași.
  - Pescarul local („angkol”) a obținut istoric o medie de **2.49 pești/zi** (în peste 400 de zile înregistrate) pescuind în siguranță în iarba de mare din apropierea țărmului.
  - Obiectivul este proiectarea unei politici autonome în `main.py` care obține cel puțin **$1.5\times$** randamentul unchiului ($\ge 3.735\text{ pești/zi}$) pe evaluările de test ascunse.

---

## 2. Mecanica Mediului și Formularea Problemei
- **Configurația Grilei Lagunei (`lagoon.py`):**
  - **Plaja Acasă `H` la $(5, 2)$:** Captura adusă la mal se contabilizează doar la atingerea `H` (golește cala). Peștii nedescărcați aflați în cală la pasul 24 se pierd.
  - **Iarba de mare `s`:** Zonă de hrănire puțin adâncă și sigură, cu probabilitate redusă de captură ($p_{\text{bite}} = 0.20$).
  - **Reciful Exterior `T` la $(0, 4)$ și $(0, 5)$:** Ape pelagice cu randament ridicat ($p_{\text{bite}} = 0.90$).
  - **Bariera de Recif `#` și Canalul `~` la $(1, 4)$:** Reciful impracticabil `#` izolează oceanul deschis, cu excepția unui singur canal `~`.
  - **Curentul Mareic:** Încheierea unui pas pe `~` declanșează un val mareic spre sud cu probabilitatea $p_{\text{tide}} = 0.40$, spălând canoea înapoi în lagună.
  - **Limita de Capacitate:** `HOLD_CAPACITY = 6` pești.
- **Spațiul Stărilor MDP și Inducția Înapoi:**
  - Sistemul este un MDP Stocastic cu Orizont Finit și probabilități de tranziție cunoscute.
  - Tuplul stării: $s = (\text{row}, \text{col}, \text{hold}, \text{steps\_left})$, unde $\text{row} \in [0..5]$, $\text{col} \in [0..6]$ (36 celule de apă), $\text{hold} \in [0..6]$ (7 niveluri de umplere) și $\text{steps\_left} \in [1..24]$.
  - Cardinalitatea totală a spațiului discret de stări:
    $$|\mathcal{S}| = 36 \times 7 \times 24 = \mathbf{6{,}048}\text{ stări}$$
  - În loc de a antrena politici RL aproximative (PPO/DQN) expuse varianței de eșantionare, politica optimă global este calculată exact prin **Iterație Înapoi a Valorii (Backward Value Iteration)**.

---

## 3. Detalii de Implementare
Pipeline-ul de rezolvare este împărțit între `solve.py` (rezolvitor exact DP), `policy_table.csv` și `main.py` (runtime de evaluare autonom):

1. **Modelarea Exactă a Tranzițiilor (`solve.py`):**
   - Reflectă fidel distribuțiile de tranziție `lagoon._advance` pentru toate cele 6 acțiuni discrete (`NORTH`, `SOUTH`, `EAST`, `WEST`, `FISH`, `WAIT`).
   - Modelează rezolvarea secvențială a mușcăturii, a mișcării dorite și a împingerii stocastice mareice.
2. **Recursiunea Înapoi Bellman:**
   - Cazul de bază: $V_0(s) = 0$.
   - Pentru pașii rămași $k = 1, \dots, 24$, rezolvă funcția optimă acțiune-valoare:
     $$Q_k(s, a) = \mathbb{E}_{\text{bite}, \text{tide}} \left[ \Delta \text{landed} + V_{k-1}(s') \right]$$
     $$V_k(s) = \max_{a \in \mathcal{A}} Q_k(s, a), \quad \pi^*(s) = \arg\max_{a \in \mathcal{A}} Q_k(s, a)$$
3. **Pachetul Autonom de Rulare (`main.py`):**
   - Precalculează toate cele 6.048 asocieri stare-acțiune și le serializează în `policy_table.csv`.
   - Înglobează o sarcină utilă de politică comprimată cu zlib și codificată base64 direct în `main.py`.
   - `main.py` rulează în câteva milisecunde fără dependențe terțe, citind `test-input.csv`, interogând dicționarul în $O(1)$ și scriind deciziile în `test-output.csv`.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 200 / 200 puncte
- **Flag Recuperat:** `SIGMOID_ELIGIBILITY`
- **Metrici de Performanță la Validare:**
  - **Randament Mediu Estimat al Agentului:** **5.968 pești/zi**;
  - **Benchmark-ul Unchiului:** **2.558 pești/zi**;
  - **Raport de Supraperformanță:** **2.333×** (depășind pragul minim de $1.50\times$).
- **Artefacte Cheie:**
  - `policy_table.csv` — tabelul complet de căutare optimă cu 6.048 intrări;
  - `main.py` — agent de producție pentru test harness;
  - `test-output.csv` — traiectorii de verificare.
