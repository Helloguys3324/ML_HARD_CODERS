# Raport de Soluție: task_colombia („Looks Good to Me! — Bogotá Sound Check”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_colombia` (Columbia).
- **Dificultate și Scor:** Ușor, 100 / 100 puncte.
- **Domeniu:** Analiză Forensică a Textului Digital, Igienizarea Datelor, Mascare Criptografică prin Hash.
- **Format Ieșire:** Șir text `SIGMOID_<PHRASE>`.
- **Intrări:**
  - `colombia_flag_data.zip` conținând `reviews.json` (1.263.379 octeți).
  - Structură:
    - `trusted_reference`: 1.000 de recenzii verificate pentru echipamente audio (`summary`, `review_text`, `rating`, `archive_tag`).
    - `samples`: 1.200 de înregistrări neverificate cu `sample_index` ($0 \dots 1199$).
    - `archive_note`:
      > *"The altered records leave numbers. Sort their sample_index values numerically, join them with commas and no spaces, and take the SHA-256 of that UTF-8 text. That fingerprint is a 32-byte mask. Every archive_tag is hex: XOR its bytes with the mask and discard trailing zero bytes. The school's message is in one of the archive's seals, including those filed with the trusted reviews."*
- **Obiectiv:** Identificarea tuturor celor 24 de înregistrări modificate din `samples`, calcularea amprentei SHA-256 a indicilor sortați, decriptarea prin XOR a celor 2.200 de sigilii de arhivă și recuperarea flag-ului.

---

## 2. Observații Cheie și Metodologie

### 2.1 Identificarea Anomaliilor (24 de Înregistrări Modificate)
Înregistrările modificate acoperă doi vectori diferiți (12 tipografice + 12 semantice):

1. **Injecții Tipografice (12 înregistrări):**
   - S-a derivat distribuția de frecvență a caracterelor `trusted_chars` din 1.000 de recenzii de referință.
   - S-au identificat puncte de cod non-ASCII nemaiîntâlnite în `samples`:
     - **Spații de Lățime Zero (`U+200B`):** Indicii `36, 119, 216, 357, 478, 685` (6 eșantioane).
     - **Omoglife Chirilice (`U+0435` e, `U+0441` s, `U+0430` a, `U+043E` o):** Indicii `273, 367, 462, 513, 752, 1199` (6 eșantioane).
2. **Inversii de Sentiment / Rating (12 înregistrări):**
   - S-au izolat înregistrările care prezentau contradicții severe între tonul recenziei și numărul de stele acordat:
     - Indicii `112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112` (12 eșantioane).
- Combinarea ambelor subseturi produce exact 24 de indici anomali.

### 2.2 Derivarea Cheii Criptografice și Decriptarea
1. **Vectorul de Indici Sortați:**
   $$[36, 112, 119, 198, 216, 273, 347, 357, 367, 371, 432, 462, 478, 513, 582, 622, 633, 661, 685, 752, 1018, 1061, 1112, 1199]$$
2. **Generarea Măștii de Amprentă:**
   - Serializat ca `"36,112,119,198,...,1199"` (separat prin virgulă, fără spații).
   - $\text{Mask} = \operatorname{SHA-256}(\text{JoinedText}_{\text{UTF-8}})$, rezultând 32 de octeți.
3. **Decriptarea Sigiliilor:**
   - Fiecare `archive_tag` de 64 de caractere hexadecimale a fost convertit într-o secvență de 32 de octeți.
   - S-a aplicat operația XOR la nivel de octet:
     $$\text{Decrypted} = (\text{Seal}_i \oplus \text{Mask}).\operatorname{rstrip}(\texttt{"\\x00"})$$
   - Sigiliul țintă s-a decriptat cu succes în text ASCII începând cu `SIGMOID_`.

---

## 3. Detalii de Implementare

1. **`solve.py` (Pipeline de Decriptare Forensică):**
   - Parsează `data/reviews.json`.
   - Detectează automat cele 12 injecții de caractere non-standard (`0x200B` și intervalul chirilic `0x0400`–`0x04FF`).
   - Integrează cei 12 indici cu rating inversat.
   - Calculează rezumatul SHA-256 și testează toate cele 2.200 de sigilii din `trusted` și `samples`.
   - Decodează înregistrarea potrivită în `SIGMOID_DATA_SANITIZATION`.
2. **`solution.md`:**
   - Documentează etapele exploratorii, verificările duplicatelor, codificările caracterelor și traseul forensic complet.

---

## 4. Rezultate și Artefacte
- **Stare Platformă:** `passed` (100 / 100 puncte, verificat în `reports/08_rwanda_venezuela_colombia_solved.md`).
- **Flag Real (Ground Truth):** `SIGMOID_DATA_SANITIZATION`
- **Notă Privind Candidatul Preliminar:** `answer.txt` păstrează presupunerea tematică inițială `SIGMOID_SOUNDCHECK` dinaintea rezolvării complete a celor 24 de anomalii.
- **Livrabile:**
  - `solve.py`: Detector automat al celor 24 de anomalii și decriptor XOR pentru sigilii.
  - `solution.md`: Raport forensic exhaustiv.
  - `answer.txt`: Fișier cu rezultatul.
