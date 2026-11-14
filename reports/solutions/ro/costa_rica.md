# Raport de Soluție: task_costa_rica („Pura Vida, No PII”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_costa_rica` (Costa Rica).
- **Dificultate și Scor:** Mediu, 200 / 200 puncte.
- **Domeniu:** Procesarea Limbajului Natural (NLP), Ingineria Confidențialității, Anonimizarea / De-identificarea Entităților Numite.
- **Context:** Agenția de turism Pura Vida Voyages partajează corespondența de asistență pentru clienți cu un serviciu terț de analiză. Toate Informațiile de Identificare Personală (PII) trebuie anonimizate/mascate, păstrând în același timp semnalele necesare pentru analizele din aval.
- **Criterii de Evaluare:**
  - **Confidențialitate (Privacy):** Proporția de segmente PII reale anonimizate/mascate (prag $\ge 0.95$).
  - **Utilitate (Utility):** Proporția de documente în care sistemul de analiză bazat pe reguli din aval extrage cu acuratețe trei atribute cheie: **subiectul (topic)**, **urgența (urgency)** și **referința rezervării (booking reference)** (prag $\ge 0.85$).
- **Format:** Script autonom `main.py` care procesează `test-input.json` (`{"documents": [{"id": ..., "text": ...}]}`) și emite `test-output.json` (`{"redacted_documents": {id: text}}`).

---

## 2. Observații Cheie și Metodologie

### 2.1 Mascare Tipizată Țintită
În locul ștergerii brute a propozițiilor sau a mascării totale, segmentele sensibile sunt înlocuite cu etichete de categorie:
- `[NAME]`: Nume de pasageri, persoane de contact și membri ai familiei.
- `[EMAIL]`: Adrese de e-mail conforme cu standardele RFC.
- `[PHONE]`: Formate telefonice internaționale și locale din Costa Rica.
- `[ADDRESS]`: Linii de stradă, bulevarde și coordonate numerotate.
- `[ID]`: Cărți naționale de identitate din Costa Rica (cédulas).
- `[DOB]`: Date explicite de naștere.
- `[ACCOUNT]`: Identificatori de conturi bancare din Costa Rica (format CR IBAN).

### 2.2 Ordine Strictă a Regulilor
Pentru a preveni coliziunile de expresii regulate (de exemplu, regulile pentru numere de telefon care pot potrivi greșit fragmente din IBAN sau numere de cédula), tiparele se execută într-o ordine strictă:
$$\text{IBAN } (\texttt{CR...}) \longrightarrow \text{Telefoane Internaționale } (\texttt{+506...}) \longrightarrow \text{Cédula} \longrightarrow \text{DOB} \longrightarrow \text{Telefoane Locale} \longrightarrow \text{Adresă} \longrightarrow \text{Email} \longrightarrow \text{Nume}$$

### 2.3 Păstrarea Utilității pentru Analize Ulterioare
- **Coduri de Rezervare:** Identificatorii ce respectă tiparul `PV-[A-Z0-9]{5}` sunt protejați explicit de mascarea numerică sau bancară.
- **Cuvinte Cheie de Clasificare:** Tokenii esențiali de intenție (`refund`, `complaint`, `urgent`, `reschedule`, `payment`, `time-sensitive`, `delay`) sunt păstrați intacți.
- **Context Geografic și de Brand:** Entitățile non-PII (`Pura Vida Voyages`, `San José`, `Manuel Antonio`, `Arenal`) sunt menținute.

---

## 3. Detalii de Implementare

1. **`main.py` (Motor de Reguli de Sine Stătător):**
   - Construit fără dependențe externe (doar modulul standard Python `re`).
   - Dicționare lexicale: `STREET_TYPES` (`Avenida`, `Calle`, `Carrera`...), `STREET_NAMES`, `MONTHS_EN`/`ES`, `SIGN_OFFS` (`Regards`, `Atentamente`...) și `RELATIONS` (`son`, `wife`, `husband`...).
   - `_extract_names(text)`:
     - Folosește normalizarea semnelor diacritice (`_ACCENT_FOLD`).
     - Extrage numele prin tipare contextuale de prezentare (*"My name is..."*, *"I am..."*).
     - Captează semnăturile de după formulele de încheiere.
     - Descompune partea locală a adresei de e-mail pentru a localiza apariția numelui în corpul textului.
   - Transformări regex:
     - IBAN: `CR\d{2}\s*(?:\d{4}\s*){4}\d{2}` / `CR\d{20}` $\rightarrow$ `[ACCOUNT]`.
     - Cédula: `\b\d-\d{4}-\d{4}\b` / `\b\d{9}\b` $\rightarrow$ `[ID]`.
     - Telefoane: `\+?506\s*[-.\s]?\d{4}[-.\s]?\d{4}` $\rightarrow$ `[PHONE]`.
     - Date de naștere: Combinații calendaristice numerice și text $\rightarrow$ `[DOB]`.
     - Adrese: Coordonate de stradă/bulevard $\rightarrow$ `[ADDRESS]`.
     - E-mailuri: Tipare RFC $\rightarrow$ `[EMAIL]`.
2. **`solve.py` (Validare pe Setul Etalon):**
   - A evaluat `main.redact` pe toate cele 20 de înregistrări de antrenare din `train-input.json`.
   - A obținut **20 / 20 potriviri exacte la nivel de caracter** cu `train-output.json`.

---

## 4. Rezultate și Artefacte
- **Stare:** `passed`
- **Scor:** `200 / 200`
- **Scor Confidențialitate:** **1.000** (prag de promovare $\ge 0.95$) ✅
- **Scor Utilitate:** **1.000** (prag de promovare $\ge 0.85$) ✅
- **Flag Recuperat:** `SIGMOID_DEIDENTIFICATION`
- **Livrabile:**
  - `main.py`: Modul de inferență gata pentru producție.
  - `solve.py`: Validator pe setul etalon și tester de regresie.
  - `writeup.md`: Documentație tehnică a definițiilor de reguli.
