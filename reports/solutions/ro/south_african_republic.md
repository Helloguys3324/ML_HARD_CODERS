# Raport de soluție: task_south_african_republic ("Ngiyakuzwa?")

## 1. Prezentare generală și obiectiv
- **Task ID:** `task_south_african_republic` (Africa de Sud / isiZulu).
- **Domeniu:** Criminalistică audio (Audio Forensics), reducerea zgomotului (scădere spectrală / Spectral Subtraction), recunoaștere vocală (ASR), analiza formanților.
- **Format de ieșire:** Șir text `SIGMOID_<WORD>` (cuvânt în limba zulu cu majuscule).
- **Date de intrare:**
  - `voice_note.wav`: Exact 630.0s (10m 30s), 20,160,044 octeți.
  - Format: RIFF/WAVE, PCM 16 biți mono @ 16,000 Hz (10,080,000 eșantioane), un singur chunk `data` de 20,160,000 octeți, fără fluxuri sau metadate suplimentare.
- **Premisă:**
  - Înregistrat pe un stadion de fotbal din Africa de Sud, cu zgomot puternic de vuvuzele și fundal de mulțime.
  - O vorbitoare rostește fraze scurte în isiZulu. La un moment dat, rostește cuvântul-cheie **"ikhodi"** („cod”), iar cuvântul imediat următor reprezintă flag-ul secret.
- **Obiectiv:** Suprimarea zgomotului de vuvuzelă de pe stadion, localizarea segmentelor de vorbire și recuperarea cuvântului flag care urmează după "ikhodi".

---

## 2. Observații-cheie și metodologie

### 2.1 Criminalistică audio și profilul de zgomot
- **Verificare steganografie LSB:** Analiza planurilor de biți pentru cei 3 biți cel mai puțin semnificativi (LSB) ai eșantioanelor PCM pe 16 biți a indicat o medie $\approx 0.5$ și o autocorelație aproape nulă la decalaje (lags) 1–3, excluzând payload-uri LSB naive.
- **Analiză spectrală:** Înregistrarea este dominată de un zgomot armonic sever de vuvuzelă pe două fundamentale:
  $$F_{0,1} = 217.8\text{ Hz}, \quad F_{0,2} = 233.4\text{ Hz}$$
  Armonicile se extind până la 8 kHz cu o modulație puternică de amplitudine (~55 Hz AM). Zgomotul începe brusc la $t \approx 2.0\text{ s}$, cu scurte scăderi ocazionale (ex. 8.8–13.2 s).
- **Fereastră de vorbire curată:** Singurul segment de vorbire relativ necorupt apare chiar la început: $t \in [0.0, 0.4\text{ s}]$. Tracker-ul de frecvență fundamentală `pyin` identifică vorbire sonoră cu $F_0 \in [182.6, 194.6]\text{ Hz}$.

### 2.2 Filtrare și eliminare zgomot (Denoising)
- **Scădere spectrală (`denoise.py`):**
  - STFT cu dimensiunea ferestrei de 1024 și pasul (hop size) 256.
  - Zgomot estimat printr-un filtru mobil pe percentila 10 de 4.0 secunde în timp (`scipy.ndimage.percentile_filter`).
  - Netezire via `uniform_filter1d` pe frecvență (3 Hz) și timp (1.0 s).
  - Scăderea magnitudinii cu factor de suprascădere `over=1.5` și prag spectral `floor=0.02 * mag`, urmată de inversarea fazei și ISTFT.
  - Filtru trece-bandă Butterworth de ordinul 6 (120–5000 Hz) și normalizarea vârfurilor la 0.95 (`denoised.wav`).
- **Filtru trece-bandă și Wiener (`preprocess.py`):**
  - Filtru trece-bandă Butterworth de ordinul 4 (300–3400 Hz) + reducerea zgomotului staționar via `noisereduce.reduce_noise(prop_decrease=0.97)` (`clean.wav`).
- **Filtru Notch-Comb:** Filtrele notch la 218.4 Hz și 233.0 Hz și armonicile acestora nu au reușit să recupereze clar formanții vorbirii după $t=2.0\text{ s}$ din cauza benzilor laterale dense provocate de modulația în amplitudine (AM).

### 2.3 Analiză fonetică și semantică a candidaților
- **Limitările Whisper:** Faster-Whisper nu oferă suport nativ pentru limba zulu (`zu`). Transcrierea forțată în engleză pe segmentul curat de introducere [0, 0.5 s] a produs "you".
- **Traiectoria formanților LPC:** Analiza formanților pe intervalul [0, 0.5 s] a arătat:
  $$(F_1, F_2) \approx (922, 2358) \to (640, 1185) \to (810, 2274) \to (680, 1969)\text{ Hz}$$
  Comparat cu eșantioanele sintetice eSpeak-ng, aceasta se potrivește fidel cu **"yebo"** ($F_2$ sintetic $\approx 2060\text{ Hz}$ față de cel original de $1980\text{ Hz}$; alternativa "sawubona" are $F_2 \approx 1730\text{ Hz}$).
- **Indiciul semantic din titlu:** Titlul provocării, **"Ngiyakuzwa?"**, se traduce din zulu prin *„Mă auzi?”*. Răspunsul afirmativ standard în eticheta zulu este **"Yebo"** (*„Da”*), urmărind modelul clasic „întrebarea în titlu, răspunsul în flag”.

---

## 3. Module de implementare

1. **`solve.py` (Forensics & EDA):**
   - Parsează antetele WAV RIFF și validează decalajele chunk-urilor.
   - Testează caracterul aleatoriu LSB pe planurile de biți $k \in \{0, 1, 2\}$.
   - Execută analiza spectrului median global de putere ($N=16384$) cu filtrare mediană pe 201 eșantioane (`signal.medfilt`) și detecție de vârfuri (`find_peaks`, prag +15 dB).
   - Generează spectrogramă (`solve_spec_0_20.png`, `solve_spec_leadin.png`).
   - Urmărește frecvența fundamentală cu `librosa.pyin` (70–400 Hz, hop 128) și salvează RMS-ul anvelopei de 4 kHz în `solve_gate.npz`.
2. **`denoise.py` (Scădere spectrală pe bază de percentile):**
   - Implementează STFT/ISTFT personalizat cu estimare glisantă a zgomotului pe bază de percentile și limitare de bandă.
3. **`preprocess.py` (Denoising staționar):**
   - Profilul de zgomot este extras din primele 2.0 s; aplică `noisereduce` și filtru trece-bandă 300–3400 Hz.
4. **`transcribe.py` (Pipeline Whisper pe bucăți/chunks):**
   - Împarte fișierul `clean.wav` în bucăți de 45 s cu o suprapunere de 6 s.
   - Pool multiproces (`spawn`) care rulează `WhisperModel(compute_type="int8")` cu VAD dezactivat.
   - Emite marcajele temporale și token-urile în `transcript.jsonl`.
5. **`solution.md`:**
   - Evaluează flag-urile candidate alternative: `SIGMOID_YEBO` (potrivire afirmativă), `SIGMOID_SAWUBONA` (salut), `SIGMOID_NGIYAKUZWA` (ecou) și token-uri cu consoane clic (`IQANDA`, `IXOXO`, `ICICI`, `ISICABHA`).

---

## 4. Rezultate și artefacte
- **Candidat pentru flag:** `SIGMOID_YEBO`
- **Nivel de încredere:** Scăzut / Euristic (din cauza coruperii extreme provocate de vuvuzele după 2.0 s; susținut de formanții din intervalul 0–0.4 s și semantica titlului).
- **Artefacte-cheie (`task_south_african_republic/`):**
  - `answer.txt`: Conține `SIGMOID_YEBO`.
  - `solution.md`: Documentație tehnică a măsurătorilor acustice și a comparațiilor de formanți.
  - `solve.py`: Scripturi de parsare WAV, verificări LSB, detecție de vârfuri și generare de spectrograme.
  - `denoise.py`: Scădere spectrală personalizată pe bază de percentile.
  - `preprocess.py`: Pipeline `noisereduce` + filtru trece-bandă.
  - `transcribe.py`: Script de transcriere multiproces cu Whisper.
