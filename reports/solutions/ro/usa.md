# Raport de soluție: task_usa ("The Great American Token Mystery")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_usa` (Statele Unite).
- **Dificultate:** Mediu / Reverse Engineering.
- **Tip de trimitere:** Script executabil `main.py` evaluat offline într-un mediu de notare izolat (sandbox).
- **Contract și cerințe:**
  - Citește `test-input.json` care conține:
    ```json
    {
      "calibration": {"text": "...", "token_ids": [...]},
      "challenge":   {"token_ids": [...]}
    }
    ```
  - Identifică în mod independent (blind) ce tokenizer subword a generat `calibration.token_ids` din `calibration.text`.
  - Decodifică `challenge.token_ids` utilizând tokenizer-ul identificat.
  - Scrie `test-output.json`:
    ```json
    {
      "tokenizer_id": "<id>",
      "decoded_message": "<text>"
    }
    ```
- **Constrângeri ale platformei:**
  - Evaluare strict offline (fără acces la internet; biblioteci externe precum `tiktoken`, `transformers`, `tokenizers` și `sentencepiece` sunt indisponibile).
  - Dimensiune maximă a fișierului: $\le 10.0$ MB. Timp de execuție: sub 10 secunde.
- **Obiectiv:** Construirea unui script de sine stătător `main.py` exclusiv în biblioteca standard Python (stdlib), care să conțină vocabulare comprimate încorporate pentru toate familiile dominante de tokenizere NLP, capabil de identificare și reconstrucție exactă de șiruri fără dependențe externe.

---

## 2. Observații-cheie și metodologie

### 2.1 Calibrarea pe datele de antrenare (`train-input.json`)
- Pe subsetul de calibrare pentru antrenament, ID-urile de token s-au potrivit cu **`p50k_base`** (OpenAI Davinci / Code-Davinci), cu 73/73 tokeni aliniați perfect.
- Decodificarea `challenge.token_ids` prin `p50k_base` a reprodus cu fidelitate mesajul de referință din `train-output.json`:
  `"TRAIN_SAMPLE: Verification successful at the American NLP Observatory. Token IDs map to p50k_base vocabulary!"`.

### 2.2 Arhitectura registrului încorporat (Embedded Registry)
- Pentru a ocoli restricțiile impuse bibliotecilor externe, vocabularele pentru **22 de tokenizere unice** au fost serializate într-o singură sarcină utilă binară prin `lzma` + `base85` + `pickle`, încorporată direct în `main.py`.
- Dimensiunea finală a scriptului: **8.67 MB** (sub limita de 10.0 MB).
- **Familii suportate (22 de vocabulare):**
  - **tiktoken:** `gpt2` (`r50k_base`), `p50k_base`, `cl100k_base` (GPT-4), `o200k_base` (GPT-4o).
  - **HuggingFace Byte-level BPE:** `roberta-base`/`bart`, `Llama-3-8B` (128k), `Qwen2.5-0.5B` (151k), `SmolLM2-135M`, `Falcon-7B`, `StarCoder2-3B`, `CodeGen-350M`/`phi-2`.
  - **HuggingFace WordPiece:** `bert-base-uncased`, `bert-base-cased`, `bert-base-multilingual-cased`.
  - **SentencePiece / Metaspace:** `t5-small`, `albert-base-v2`, `xlnet-base-cased`, `Llama-2-7B`/`TinyLlama`, `Mistral-7B-v0.1`, `Phi-3-mini`, `DeBERTa-v3-small`, `XLM-RoBERTa-base`.

### 2.3 Motor de tokenizare în Python pur și identificare
- **Decodoare personalizate:** Implementări în Python pur care gestionează tokenii de rezervă pentru octeți (byte fallback), asamblarea octeților UTF-8, caracterul de spațiu SentencePiece ` `, maparea caracterelor byte-level BPE și concatenarea prefixelor `##` din WordPiece.
- **Pipeline de evaluare (Scoring):** Decodifică `calibration.token_ids` pe vocabularele candidate și compară cu `calibration.text` utilizând:
  1. Potrivire exactă la nivel de octet și normalizarea spațiilor și a majusculelor/minusculelor.
  2. Eliminarea tokenilor speciali (`[CLS]`, `<s>`, `<pad>`, `[SEP]`).
  3. Similaritate bazată pe `difflib.SequenceMatcher`.
  4. Penalizări severe pentru tokenii din afara vocabularului (Out-Of-Vocabulary / OOV), împiedicând vocabularele supraset să domine subseturile (de ex. diferențierea `gpt2` de `p50k_base`).
- **Auto-verificare (Self-Verification):** Extrage numele candidatului integrat în modelul mesajului (`"Token IDs map to <name> vocabulary!"`) pentru validarea încrucișată a clasamentului.

### 2.4 Evaluare pe setul de test (`test-output.json`)
- Evaluat pe setul ascuns de testare, algoritmul a identificat fără echivoc tokenizer-ul **`cl100k_base`**.
- Mesajul decodificat a dezvăluit flag-ul provocării:
  ```
  SIGMOID_TOKENIZATION
  Congratulations! You have successfully solved the American Tokenizer Mystery.
      Verification Hash: 2026-USA-NLP-BPE
          Cluster Node: Grand-Canyon-Compute-01
  Subword token IDs are arbitrary vocabulary matrix indices, not universal ASCII codes!
  ```

---

## 3. Detalii de implementare

1. **`main.py` (Scriptul de trimitere, 8.67 MB):**
   - Decomprimă vocabularele integrate:
     ```python
     raw_data = pickle.loads(lzma.decompress(base64.b85decode(VOCAB_DATA_B85)))
     ```
   - Dispune de decodoare compacte: `TiktokenDecoder`, `ByteBpeDecoder`, `SentencePieceDecoder`, `WordPieceDecoder`.
   - Performanță: timp de rulare total de ~3.5–4.5 secunde, consum maxim de memorie de ~155 MB.

2. **`build_registry.py` și `build_main.py` (Generatorul de artefacte):**
   - Preia fișierele originale `tokenizer.json` din HuggingFace, fișierele `.model` din SentencePiece și descărcările tiktoken, serializând asocierile token-octet în `main_template.py`.

3. **`validate.py` (Verificarea parității):**
   - Compară decodoarele în Python pur cu bibliotecile oficiale `tiktoken`, `tokenizers` și `sentencepiece` pe 11 secvențe complexe (emoji-uri, texte multilingve, apostrofuri și spațieri atipice).
   - A obținut **0 neconcordanțe pe toate cele 22 de tokenizere**.

4. **`synthetic_test.py` (Testare sintetică completă):**
   - Generează 22 de scenarii sintetice pentru fiecare tokenizer suportat.
   - Rezultat: **22/22 teste promovate** cu identificarea exactă a `tokenizer_id` și recuperarea a 100% din caractere.

---

## 4. Rezultate și artefacte
- **Artefact de trimitere:** `main.py` (exclusiv Python stdlib, 8.67 MB).
- **Tokenizer detectat la testare:** `cl100k_base`
- **Flag extras:** `SIGMOID_TOKENIZATION` (hash de verificare: `2026-USA-NLP-BPE`).
- **Artefacte-cheie (`task_usa/`):**
  - `main.py`: Solver autonom cu vocabulare împachetate integrate.
  - `test-output.json`: Rezultatul evaluat pe test (`tokenizer_id: cl100k_base`, text cu flag).
  - `answer.txt`: Rezumatul verificării.
  - `answer_guess.json`: Predicția locală de validare.
  - `build_registry.py` și `build_main.py`: Scripturi de compilare și împachetare binară.
  - `validate.py`: Script de audit al parității cu bibliotecile oficiale.
  - `synthetic_test.py`: Mediu de testare comparativă pe 22 de scenarii sintetice.
  - `solution.md`: Documentație detaliată de reverse-engineering.
