# Raport de Soluție: task_iceland („What The Data?”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Ușoară (Easy, 100 pct). Biroul Meteorologic din Islanda monitorizează izvoarele geotermale calde (Geysir, Mývatn etc.). Transmisiunile prin satelit ale fișierului `test-input.csv` suferă de corupere sistemică severă:
  1. *Zgomot în Antet (Preamble Noise):* Comentarii introductive (`#`) și linii goale arbitrare.
  2. *Mojibake Multivelar (Multi-Level Mojibake):* Caractere islandeze UTF-8 decodificate eronat în mod repetat ca Latin-1 / Windows-1252 pe o adâncime de 1 până la 3 niveluri recursive (de ex., `MÃ½vatn` $\to$ `Mývatn`, `HveragerÃ°i` $\to$ `Hveragerði`).
  3. *Formatarea Numerelor Zecimale:* Separatori numerici inconsecvenți (virgule europene vs puncte americane: `83,7` vs `83.7`).
  4. *Inconsecvența Unităților de Măsură:* Unități mixte de grade Celsius (`C`) și Fahrenheit (`F`) în coloana `unit`.
  5. *Duplicare la Retransmisie:* Înregistrări `reading_id` duplicate cu unități de măsură conflictuale sau denumiri de stații corupte.
- **Obiectiv:** Script autonom `main.py` care procesează `test-input.csv` într-un fișier curățat `test-output.csv`:
  ```csv
  reading_id,station_name,temperature_c,reading_date
  ```
  Sortat strict după `reading_id` întreg, cu temperaturi în Celsius formatate cu o singură zecimală (`{:.1f}`) și toponime islandeze restaurate impecabil.

## 2. Metodologie Cheie
Un pipeline deterministic bazat exclusiv pe biblioteca standard (`csv`, `sys`), fără dependențe grele:
1. **Eliminarea Preambulului și Identificarea Antetului:** Citire cu codificarea `utf-8-sig` (gestionând potențiale caractere BOM), ignorarea comentariilor care încep cu `#` și a liniilor goale, precum și parsarea antetelor delimitate prin punct și virgulă (`;`).
2. **Repararea Recursivă a Erorilor de Codificare / Mojibake (`fix_station_name`):** Testează recursiv re-codificarea prin `latin-1`, `cp1252` și `iso-8859-15` urmată de decodificare în `utf-8` până la convergență. Caracterele islandeze autentice (`ð`, `þ`, `æ`, `ö`, `í`, `ý` etc.) generează excepții la re-codificare și rămân protejate.
3. **Unificarea Unităților:** Înlocuiește virgulele cu puncte și convertește gradele Fahrenheit în grade Celsius:
   $$T_C = (T_F - 32.0) \times \frac{5.0}{9.0}$$
4. **Deduplicare și Rezolvarea Conflictelor:** Pentru înregistrările cu `reading_id` repetat, păstrează citirile native în Celsius pentru a elimina erorile de rotunjire la conversia inversă și alege șirul stației cu cel mai mic număr de markeri de corupere (`Ã`, `\ufffd`).
5. **Export Determinist:** Sortează după `reading_id` întreg și formatează temperatura ca `{:.1f}`.

## 3. Detalii de Implementare (`main.py`)
- **Decodor Recursiv de Mojibake:**
  ```python
  def fix_station_name(name: str) -> str:
      if not name:
          return ""
      name = name.strip(' "\'\t\r\n')
      for enc in ("latin-1", "cp1252", "iso-8859-15"):
          try:
              candidate = name.encode(enc).decode("utf-8")
              if candidate != name and candidate:
                  return fix_station_name(candidate)
          except Exception:
              pass
      return name
  ```
- **Pipeline de Curățare (`clean_export`):**
  Folosește `csv.DictReader` pe liniile filtrate cu delimitatorul `;`, normalizează datele calendaristice, temperaturile și numele stațiilor, rezolvând intrările duplicate printr-un dicționar înainte de sortare și salvare pe disc.

## 4. Rezultate și Artefacte
- **Validare:**
  - *Date de Referință:* Potrivire octet cu octet cu fișierul `train-output.csv` (hash SHA-256 identic: `5b3c5197...07aa1f`).
  - *Suită de Testare la Stres:* Recuperare de 100% pe 120 de înregistrări sintetice puternic degradate (mojibake pe 3 niveluri, unități de măsură mixte).
- **Flag:** `SIGMOID_PREPROCESSING`.
- **Fișiere:** `main.py`, `solution.md`, `stress_test.py`.
