# Raport de soluție: task_united_kingdom ("Call My BLEUff")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_united_kingdom` (Regatul Unit).
- **Dificultate:** Ușor / Mediu.
- **Domeniu:** Manipularea metricilor NLG (NLG Metrics Gaming) / NLP Forensics + Decodificare steganografică ternară (Baza 3).
- **Format de ieșire:** Șir text `SIGMOID_<WORD>` (cuvânt englezesc din 8 litere).
- **Date de intrare:**
  - `references.csv`: 24 de runde $\times$ 40 carduri de restaurante $\times$ 5 descrieri de referință = 4,800 rânduri.
  - `panel.csv`: 24 de runde $\times$ 40 carduri $\times$ 3 locuri în juriu (seat 1, seat 2, seat 3) = 2,880 descrieri. Un „pachet” (deck) reprezintă setul de 40 de carduri propus de un membru într-o rundă.
  - `bleu.py`: BLEU-4 la nivel de corpus, fără netezire, cu litere mici și penalizare de lungime (brevity penalty) raportată la cea mai apropiată lungime de referință.
- **Premisa și paradoxul BLEU:**
  - Inspirat din emisiunea clasică britanică de televiziune *Call My Bluff*. În fiecare rundă, trei jurați propun descrieri: doi trișează exploatând metrica BLEU, în timp ce unul este un narator onest.
  - Selectarea pachetului cu cel mai mare scor BLEU în fiecare rundă produce șirul agramat `SKELDRAP`. Deoarece BLEU standard măsoară exclusiv precizia n-gramelor, acesta recompensează masiv trunchierea adversarială, umplutura (padding-ul), îmbinarea forțată și copierea verbatim.
- **Obiectiv:** Crearea unei metrici de penalizare a „necinstei” (anti-gaming penalty) pentru a identifica juratul onest în toate cele 24 de runde și a decodifica cuvântul englezesc de 8 litere ascuns.

---

## 2. Observații-cheie și metodologie

### 2.1 Anatomia tacticilor de manipulare a metricii BLEU
Analiza pachetelor cu scoruri BLEU ridicate a evidențiat patru tipare distincte de manipulare:
1. **Plagiat verbatim / concatenare:** Fuzionarea a două propoziții complete de referință ($p_4 \approx 1.0$), generând artificial scoruri BLEU de 0.74–0.78 (până la 62 de carduri copiate per pachet).
2. **Padding și duplicare a faptelor:** Repetarea redundantă a atributelor pentru a mări suprapunerea n-gramelor și a contracara penalizarea de scurtare.
3. **Trunchiere agresivă:** Tăierea propozițiilor la mijlocul clauzei („It is a kids.”, „This pub.”), maximizând precizia n-gramelor scurte și absorbind penalizări moderate de lungime la nivel de corpus.
4. **Îmbinare forțată (Splicing):** Îmbinarea unor clauze fragmentate în fraze lipsite de logică gramaticală („A high end family it a 3 out of 5 customer rating...”).

### 2.2 Metrica penalizării „necinstei” pe partea receptorului
Pentru fiecare din cele 2,880 de descrieri generate, au fost calculate penalizări pe baza unor indicatori ignorați de BLEU:
- **Copiere verbatim:** $p_4 \ge 0.8$ sau similaritate `difflib.SequenceMatcher` $\ge 0.95$.
- **Plagiat:** $\ge 85\%$ din 5-grame prezente în propozițiile de referință.
- **Distorsiune a lungimii:** Raportul lungimii față de cea mai apropiată referință $< 0.72$ (trunchiere) sau $> 1.30$ (padding).
- **Omisiune / Recall scăzut:** Reamintirea informației $< 0.55$ pe 5 tipuri de sloturi (preț, rating, prietenos cu familiile, zonă, specific culinar).
- **Trunchiere suspendată:** Prepoziții sau conjuncții lăsate la final (`with`, `in`, `to`, `at`).
- **Halucinații:** Contrazicerea directă a faptelor din referințe.
Scorul total de neconformitate al unui pachet este suma penalizărilor celor 40 de carduri; juratul onest minimizează acest scor.

### 2.3 Decodificarea ternară (Baza 3) a cuvântului
- Cele 24 de runde corespund la 8 triplete de câte 3 runde (rundele 1–3, 4–6, ..., 22–24), fiecare codificând o literă.
- În fiecare rundă, juratul onest $s \in \{1, 2, 3\}$ reprezintă o cifră ternară $d = s - 1 \in \{0, 1, 2\}$.
- Valoarea numerică a literei este calculată astfel:
  $$\text{valoare} = d_0 \cdot 9 + d_1 \cdot 3 + d_2 \quad (0 \to \text{'A'}, \dots, 25 \to \text{'Z'})$$
- Selecția simplă de tip greedy pe baza penalizărilor minime a returnat șirul zgomotos `LLBKENRN`.
- **Validarea cu dicționarul SCOWL:** Constrângerea secvenței pe baza tuturor celor 11,526 de cuvinte de 8 litere din dicționarul britanic SCOWL (`en_GB-large.dic`) a identificat **`NORTHERN`** drept candidatul cu penalizarea minimă globală (scor: **218.5**):
  - **N** (13) $\to (1, 1, 1) \to$ Rundele 1–3: Jurații (2, 2, 2)
  - **O** (14) $\to (1, 1, 2) \to$ Rundele 4–6: Jurații (2, 2, 3)
  - **R** (17) $\to (1, 2, 2) \to$ Rundele 7–9: Jurații (2, 3, 3)
  - **T** (19) $\to (2, 0, 1) \to$ Rundele 10–12: Jurații (3, 1, 2)
  - **H** (7)  $\to (0, 2, 1) \to$ Rundele 13–15: Jurații (1, 3, 2)
  - **E** (4)  $\to (0, 1, 1) \to$ Rundele 16–18: Jurații (1, 2, 2)
  - **R** (17) $\to (1, 2, 2) \to$ Rundele 19–21: Jurații (2, 3, 3)
  - **N** (13) $\to (1, 1, 1) \to$ Rundele 22–24: Jurații (2, 2, 2)
- Clasate pe locurile următoare: `FOLKMOOT` (221.5), `TURNDOWN` (222.8), `LLEWELYN` (229.4).

---

## 3. Detalii de implementare

1. **`solve.py` (Pipeline reproductibil de analiză și decodificare):**
   - Parsează `references.csv` și `panel.csv`.
   - Calculează BLEU-4 de bază per pachet, evidențiind cuvântul-capcană `SKELDRAP`.
   - Evaluează matricea de penalizări la nivel de card ($24 \times 3$).
   - Notează cuvintele candidate din dicționar prin maparea ternară a locurilor și emite `SIGMOID_NORTHERN`.

2. **`solution.md`:**
   - Detaliere pas cu pas a penalizărilor per rundă și a tacticilor de trișare identificate.
   - Demonstrație matematică a schemei de codificare în baza 3 și a ierarhizării din dicționar.

---

## 4. Rezultate și artefacte
- **Candidat pentru flag:** `SIGMOID_NORTHERN`
- **Nivel de încredere:** Mediu / Euristic (cea mai bună potrivire dintre 11,526 de intrări din dicționarul britanic cu penalizare minimă cumulată, eliminând complet artefactele de manipulare BLEU).
- **Artefacte-cheie (`task_united_kingdom/`):**
  - `answer.txt`: Conține `SIGMOID_NORTHERN`.
  - `solve.py`: Script automatizat pentru audit BLEU, calcul de penalizări și decodificare din dicționar.
  - `solution.md`: Raport tehnic cu descrierea mecanismelor de manipulare, a matricelor de penalizări și a comparațiilor de candidați.
