# Raport de Soluție: task_cuba („If you know, you know! — Zemí”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_cuba` (Cuba).
- **Dificultate și Scor:** Dificil, 300 / 300 puncte.
- **Domeniu:** Regăsirea Informației (Information Retrieval, IR), Căutare Hibridă de Pasaje de Înaltă Performanță, Compresie Extremă Vectorială/Lexicală.
- **Corpus:** 1.552 de documente diverse (470 de capitole de carte, 955 de articole Wikipedia, 105 rapoarte, 14 scanări OCR degradate, 8 tabele) interogate prin parafrazări descriptive ample.
- **Constrângeri Inginerești ale Platformei (Toate 4 Obligatorii):**
  1. **Dimensiunea Indexului Necomprimat (`main.py` + `index/`):** $\le 53.0$ MB (referință inițială: 80.25 MB — EȘEC).
  2. **Latența la Pornire la Rece (`Retriever.__init__`):** $\le 10.0$ secunde.
  3. **Latența de Inferență (p95 per interogare):** $\le 75.0$ milisecunde.
  4. **Precizia Regăsirii ($\text{Hit@3}$):** $\ge 0.80$ (referință inițială: 0.709 — EȘEC).
- **Format Trimitere:** Arhivă `task_cuba.zip` conținând `main.py` și directorul serializat `index/`.

---

## 2. Observații Cheie și Metodologie

### 2.1 Regăsire Hibridă cu Granularitate Multiplă
Reprezentările dense pure (`model2vec`) au avut dificultăți la regăsirea cuvintelor cheie rare ($\text{Hit@3} = 0.578$), în timp ce BM25 global a eșuat pe parafrazări subtile. S-a conceput o ierarhie de regăsire pe trei niveluri:
1. **BM25 Lexical la Nivel de Document (`bm25s`):**
   - Formula Lucene ($k_1=1.5, b=0.75$) cu multiplicarea de $3\times$ a titlului pentru prioritizarea alinierii tematice ($\text{Hit@3} = 0.806$).
2. **BM25 Lexical la Nivel de Pasaj (Chunk):**
   - Ferestre glisante de 350 de cuvinte cu suprapuneri de 100 de cuvinte (16.246 fragmente în total). Păstrează scorul maxim al fragmentului per document părinte pentru localizarea pasajelor restrânse ($\text{Hit@3} = 0.787$).
3. **Regăsire Densă la Nivel de Pasaj (`model2vec`):**
   - Fragmente de 100 de cuvinte (pas de 50 de cuvinte, 26.354 vectori) încapsulate prin modelul static pe 256 de dimensiuni `potion-base-8M` din fișierele sarcinii (fără acces la rețea).

### 2.2 Reierarhizator Logistic Supervizat (Supervised Logistic Reranker)
- Antrenat pe 268 de interogări de dezvoltare prin regresie logistică cu regularizare $L_2$ ($C=0.03$) pe un vector de 16 caracteristici:
  - Scoruri z intra-interogare pentru BM25 Document, BM25 Pasaj și similitudine Densă (standardizând variațiile între lungimile interogărilor).
  - Scoruri brute și diferențe față de scorul maxim ($\text{max\_score} - \text{score}$).
  - Metrici ale lungimii documentelor ($\log(\text{len})$ și numărul brut de tokeni).
  - Indicatori one-hot pentru tipul sursei documentului (cărți, wiki, rapoarte, OCR, tabele).
- Validarea încrucișată (`GroupKFold` pe interogări) a atins **0.866 Hit@3** (depășind fuziunea euristică Reciprocal Rank Fusion, situată la 0.851).

### 2.3 Cuantificare Extremă și Optimizare a Dimensiunii (Redusă la 41.18 MB)
- **Ponderi BM25:** Conversia matricelor rare de scor din `float32` în **`float16`** fără inversarea clasamentului ($14.2 \to 7.1$ MB).
- **Cuantificarea Vectorilor Deși:** Cuantificarea embedding-urilor de pasaje pe 256 de dimensiuni în **`int8`** cu factori de scalare pe coordonate ($26.4 \to 6.6$ MB).
- **Cuantificarea Codificatorului:** Cuantificarea ponderilor modelului `model2vec` în **`int8`** ($29 \to 7.6$ MB; corelație cosinus $r = 0.9996$).
- **Curățarea Mediului de Rulare:** PyTorch a fost complet eliminat din fluxul de inferență, operând exclusiv cu extensii native C (`numpy`, `bm25s`).

---

## 3. Detalii de Implementare

1. **`build_index.py` (Constructor de Index și Cuantificator):**
   - Procesează corpusul brut, construiește indicii BM25 pentru documente și fragmente, convertește la `float16`, generează tabelele dense în `int8` și exportă `lr_model.json`.
2. **`submission/main.py` (Retriever pentru Inferență):**
   - `Retriever.__init__`: Încarcă structurile `bm25s`, tabelele cuantificate și coeficienții liniari în $1.89$ s (limita fiind de 10.0 s).
   - `Retriever.search(query, top_k=3)`: Execută interogări lexicale paralele, calculează produsele scalare dense prin multiplicare matriceală, construiește 16 caracteristici și aplică reierarhizatorul liniar în $\approx 27.5$ ms (p95).
3. **`solve.py`:**
   - Automatizează generarea indexului, validează integritatea directoarelor și generează `task_cuba.zip` (`compresslevel=9`).

---

## 4. Rezultate și Artefacte

### Teste de Evaluare Comparativă (`local_check.py` pe Arhiva Finală)
| Metrică | Observat | Constrângere | Verdict |
|:---|:---:|:---:|:---:|
| **Dimensiunea Indexului Necomprimat** | **41.18 MB** | $\le 53.0$ MB | **PASS** |
| **Timp de Inițializare la Rece** | **1.89 s** | $\le 10.0$ s | **PASS** |
| **Latență de Căutare p95** | **27.5 ms** | $\le 75.0$ ms | **PASS** |
| **Precizie ($\text{Hit@3}$)** | **0.858** (CV: 0.866) | $\ge 0.80$ | **PASS** |
| **Precizie Top-1 / Top-10** | **0.694** / **0.937** | — | Diagnostic |

- **Verdictul Verificării:** `ALL LIMITS PASSED`
- **Livrabile:**
  - `task_cuba.zip`: Pachet verificat pentru platformă (25.5 MB comprimat, 41.18 MB necomprimat).
  - `answer.txt`: Descriptor de trimitere care indică spre `task_cuba.zip`.
  - `build_index.py`: Pipeline de generare și cuantificare a indexului.
  - `solve.py`: Script automatizat pentru construirea arhivei.
  - `solution.md`: Documentație inginerească completă și studii de ablație.
