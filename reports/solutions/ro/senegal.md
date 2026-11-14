# Raport de Soluție: task_senegal ("The Teranga FM Token Diet")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_senegal` (Senegal).
- **Dificultate și Scor:** Medium, 200 / 200 puncte.
- **Domeniu:** General AI / Optimizarea Costurilor LLM și Rutare Dinamică a Interogărilor.
- **Format Ieșire:** Tabel de rutare `routing_config.json` și flag `SIGMOID_<NAME>`.
- **Premisă:**
  - Postul comunitar Teranga FM din Kaolack (bazinul arahidelor din Senegal) operează o linie de asistență prin SMS pentru fermieri, susținută de un API de inferență LLM.
  - Implementarea de referință ruta toate interogările primite către un model de nivel înalt (high-tier) cu stil detaliat (verbose), exemple few-shot și context complet netrunchiat, consumând 2.277 de tokeni per lot de evaluare.
  - Obiectivul este construirea unui tabel de rutare optim (`routing_config.json`) pentru 20 de interogări de referință din `fact_bank.json`, menținând acuratețea QA de 100% (20/20 `PASS`) sub o limită strictă de maximum **1.200 de tokeni în total** (o reducere de ~49%).

---

## 2. Perspective Cheie și Mecanica Rutării
- **Logica Simulatorului și a Verificării (`model_utils.py`):**
  - **Criteriul de Reușită (`question_passes`):** O completare satisface o întrebare dacă și numai dacă fiecare slot factual obligatoriu conține cel puțin o expresie validă din `slot["phrasings"]` identificată ca subșir.
  - **Condiția de Includere a Slotului (`render_completion_body`):**
    $$\text{assisted} \lor \neg \text{slot}[\text{"large\_only"}], \quad \text{unde } \text{assisted} = (\text{tier} == \text{"large"}) \lor \text{context}$$
  - **Influența Parametrilor Asupra Costului:**
    - `fewshot`: Introduce un overhead substanțial în prompt fără a schimba eligibilitatea sloturilor $\implies$ `fewshot = False` pentru toate apelurile.
    - `style`: `verbose` adaugă fraze introductive și de încheiere standard; `terse` generează text factual compact $\implies$ `style = "terse"` uniform.
    - `context`: Fragmentele de context RAG sunt necesare doar pentru nivelul `cheap` la procesarea sloturilor `large_only`. Pe nivelul `large`, contextul este redundant.
- **Politica Optimă de Rutare:**
  - **Întrebări fără sloturi `large_only` (8 interogări):** Alocate nivelului `cheap` fără context (`cheap / terse / fewshot=False / context=False`).
  - **Întrebări cu sloturi `large_only` (12 interogări):** Compromis de cost: se alege `cheap + context=True` dacă costul în tokeni al fragmentului de context este mai mic decât diferența de cost pentru `large + context=False`.

---

## 3. Detalii de Implementare
Pipeline-ul de optimizare este coordonat prin `solve.py` și validat în `main.py`:

1. **Căutare pe Grila de Parametri (`solve.py`):**
   - Pentru toate cele 20 de întrebări, evaluează toate cele 16 configurații din produsul cartezian:
     $$\text{tier} \in \{\text{"cheap"}, \text{"large"}\}, \; \text{style} \in \{\text{"terse"}, \text{"verbose"}\}, \; \text{fewshot} \in \{\text{False}, \text{True}\}, \; \text{context} \in \{\text{False}, \text{True}\}$$
   - Simulează ieșirile prin `mu.build_call(...)` și validează `mu.question_passes(...)`.
2. **Minimizarea Costului cu Constrângeri:**
   - Filtrează combinațiile eșuate și reține configurația cu consum minim de tokeni per interogare:
     $$c^*(q) = \arg\min_{c \in \mathcal{C}_{\text{pass}}(q)} \text{Tokens}(q, c)$$
   - *Grupul A (8 interogări: `q01`, `q03`, `q09`, `q11`, `q14`, `q15`, `q16`, `q19`):* Rutate către `cheap / terse / fewshot=False / context=False`.
   - *Grupul B (12 interogări cu sloturi `large_only`):* Rutează dinamic între `cheap + context` și `large` în funcție de lungimea în tokeni a fragmentului de context.
3. **Execuție și Pachetare (`main.py`):**
   - Exportă tabelul ales în `routing_config.json`.
   - Rulează execuția completă pe lotul `fact_bank.json`, scriind răspunsurile validate în `answers.json` și generând `submission.json`.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 200 / 200 puncte
- **Flag Recuperat:** `SIGMOID_TOKENDIET`
- **Metrici de Resurse:**
  - **Acuratețe:** **20 / 20 întrebări trecute cu succes (100.0%)**;
  - **Consum de Tokeni:** **1.160 tokeni** (față de limita de 1.200 și nivelul inițial de 2.277, obținând o **reducere de 49.1%**).
- **Artefacte Rezultate:**
  - `routing_config.json` — matricea optimă de dispecerizare per interogare;
  - `answers.json` — completările răspunsurilor generate de model;
  - `submission.json` — sarcina utilă oficială pentru trimiterea la benchmark.
