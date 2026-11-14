# Raport de Soluție: task_egypt („Always Valid, No Excuses”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_egypt` (Egipt).
- **Dificultate și Scor:** Ușor, 100 / 100 puncte.
- **Domeniu:** Parsare de Text Structurat, Extracția Zgomotului OCR, Igienizarea Deterministă a Datelor.
- **Date de Intrare:** `test-input.txt` (înregistrări delimitate pe linii, codificate în UTF-8). Fiecare linie conține text degradat rezultat din recunoașterea optică a caracterelor (OCR), reprezentând o înregistrare din arhivele egiptologice.
- **Specificații de Normalizare a Datelor:**
  1. *Curățarea Segmentelor:* Delimitate prin caracterul bară verticală `|`; spațiile albe de la început și sfârșit trebuie eliminate din chei și valori.
  2. *Suprimarea Artefactelor:* Zgomotul de scanare trebuie filtrat: markeri de pagină (`page 22`), titluri de capitole (`Chapter IX`), cratime de paginație (`- 45 -`), note de defecte în paranteze drepte (`[torn]`, `[illegible]`, `[stain]`), etichete de ilustrații (`Fig 3`, `Plate VII`, `footnote 3`) și delimitatori aleatorii (`###`, `~~~`).
  3. *Validarea Cheilor:* Un segment este recunoscut ca un câmp valid dacă și numai dacă respectă sintaxa `key: value`, iar cheia respectă expresia regulată `^[a-zA-Z_][a-zA-Z0-9_]*$`.
  4. *Conversia Tipului de Date:*
     - Valori goale $\rightarrow$ JSON `null` (`None` în Python).
     - Șirurile exacte `"true"` / `"false"` $\rightarrow$ booleeni JSON `true` / `false`.
     - Numere zecimale (de ex., `40`, `0`, `-1264`) $\rightarrow$ numere întregi native JSON.
     - Toate celelalte intrări $\rightarrow$ șiruri JSON (păstrând ghilimelele și slash-urile escaped).
     - Împărțirea se execută strict după **primul** caracter două puncte `:` (de ex., `ratio: 3:1` $\rightarrow$ `"ratio": "3:1"`).
  5. *Invariant de Ieșire:* Exact un singur obiect JSON strict formatat per linie scris în `test-output.txt`. Încălcările de format sau discrepanțele în numărul de linii conduc la respingerea soluției.

---

## 2. Observații Cheie și Metodologie

### 2.1 Motor Determinist de Reguli vs. Inferență LLM
Deși descrierea sarcinii sugera un oracol opțional pe bază de LLM local (Qwen3-1.7B-GGUF), modelele generative de limbaj introduc cazuri limită nedeterministe: ghilimele halucinate în jurul numerelor întregi, câmpuri rare omise, secvențe de escape corupte pe căile de fișiere și riscuri de timeout pe CPU. În schimb, a fost proiectat un motor determinist de expresii regulate bazat exclusiv pe biblioteca standard Python, garantând zero halucinații și procesare în microsecunde.

### 2.2 Filtrarea Zgomotului prin Validarea Identificatorilor
- Împărțirea liniilor după `|` și filtrarea segmentelor care nu conțin `:` elimină fragmentele brute de text.
- Separarea la primul două puncte (`seg.split(":", 1)`) și verificarea `^[a-zA-Z_][a-zA-Z0-9_]*$` pentru `key.strip()` elimină automat segmentele complexe de zgomot care conțin două puncte (de ex., `Chapter IX: The Tomb`, `page 22: note`, `[torn]: missing`).

### 2.3 Serializare Strictă în JSON
- Înregistrările sunt serializate folosind `json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))`.
- Impunerea parametrului `ensure_ascii=True` efectuează escape pentru caracterele de control și trecerile la linie nouă, prevenind coruperea înregistrărilor pe mai multe linii.
- Validarea ciclului complet de deserializare prin `json.loads` este verificată în memorie înainte de scrierea fișierului.

---

## 3. Detalii de Implementare

1. **`main.py` (Parser de Producție):**
   - **`parse_record(line: str) -> dict`**:
     - Elimină caracterele de linie nouă și împarte după `|`.
     - Validează structura cheii prin `^[a-zA-Z_][a-zA-Z0-9_]*$`.
     - Parsează numere întregi prin `^-?(?:0|[1-9]\d*)$`, mapează valorile booleene și null și convertește restul conținutului în șiruri text.
   - **`find_input_file(cwd, script_dir) -> Path`**:
     - Rezolvă căile prin argumente CLI, fișiere din directorul curent de lucru (`test-input.txt`, `input.txt`) și semnături de conținut (`|` și `:`).
   - **`main()`**:
     - Procesează înregistrările de intrare linie cu linie și scrie obiectele JSON validate în flux direct către `test-output.txt`.
2. **`validate.py` (Suită de Regresie):**
   - Validează datele de probă pe `train-output.txt` (13/13 potriviri structurale exacte JSON, parsând corect căile Windows `C:\papyri\thoth`, ghilimelele interioare `"lord of the moon"` și numerele negative).
   - Validează cazul limită din specificații:
     `page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX`
     $\implies$ `{"king":"Menes","title":"ruled with \"the Aten\"","alive":true}`.
   - Asigură zero abateri de la schemă pentru șiruri vide, caractere două puncte încorporate și limite numerice (`ALL OK`).
3. **`main.py.txt` (Fișier de Rezervă pentru Platformă):**
   - Copie identică octet cu octet (MD5 `c6693a0ce13bfd3abd6304fa431495b8`) pentru sistemele de încărcare ale platformei care solicită extensia `.txt`.

---

## 4. Rezultate și Artefacte
- **Verdictul Validării:** `ALL OK` (`[sample] OK (13 lines)`, `[spec-example] OK`, `[edge] OK`).
- **Capacitate de Procesare:** 100 de înregistrări de evaluare parsate în **0.44 secunde** pe CPU (limita permisă: 60.0 s).
- **Livrabile:**
  - `main.py`: Motor de parsare complet autonom, fără dependențe externe.
  - `main.py.txt`: Fișier oglindă secundar pentru compatibilitatea cu cerințele platformei.
  - `validate.py`: Cadru de testare pentru probe de referință și cazuri limită.
  - `test-output.txt`: Fișier de ieșire etalon validat.
  - `solution.md`: Raport exhaustiv de arhitectură și specificații.
  - `answer.txt`: Instrucțiuni de trimitere pentru execuție (`SUBMIT_FILE ...`).
