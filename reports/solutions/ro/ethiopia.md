# Raport de Soluție: task_ethiopia („The Azmari's Borrowed Melody”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Medie (Medium, 200 pct). Înregistrările solo de arhivă ale muzicienilor azmari din Etiopia (lăuta masenqo sau flautul washint) conțin o îmbinare audio perfectă (splice), inserată dintr-o altă dublă, fără variații audibile de volum sau artefacte de tip clic.
- **Obiectiv:** Detectarea limitelor temporale exacte (`start_s`, `end_s` în secunde) ale fragmentului inserat în 3 înregistrări de test ascunse (`test-1`, `test-2`, `test-3`).
- **Format de Trimitere și Protocol de Testare:**
  - Script independent `main.py` executat prin: `python main.py <input.wav>`.
  - Generează fișierul `detection.csv` în directorul curent de lucru:
    ```csv
    start_s,end_s
    00.0,00.0
    ```
  - Criterii de succes: Erori absolute ale limitelor $|start - true\_start| \le 1.5\,\text{s}$ și $|end - true\_end| \le 1.5\,\text{s}$ simultan pe toate cele 3 fișiere de test.

## 2. Perspectivă Cheie și Metodologie
Deși volumul RMS și tranzițiile au fost armonizate în timpul montajului (splicing), funcția de transfer acustic a mediului de înregistrare și profilurile de zgomot de fond (noise floor) diferă între duble.

1. **Profilare Spectrală:** Transformata Fourier pe Timp Scurt (STFT) extrasă pe 24 de benzi de frecvență spațiate logaritmic (de la 40 Hz la $0.45 \cdot f_s$). Caracteristicile per bandă includ puterea totală ($E$) și puterea la percentila 20 ($F$).
2. **Standardizare Robustă:** Log-energiile spectrale per cadru sunt centrate în raport cu media cadrului (anulând fluctuațiile globale de volum) și standardizate utilizând mediana și Deviația Mediană Absolută (MAD), trunchiate la $[-12, 12]$. Caracteristicile sunt concatenate în matricea $Z \in \mathbb{R}^{N_{\text{frames}} \times 46}$.
3. **Scanarea Segmentelor prin Sume Cumulative (`scan_segment`):**
   - Scanează segmentele candidate $[i, j]$ (cu o durată între 1.2s și 25s) ce au o pondere a cadrelor active $\ge 60\%$.
   - Maximizează distanța euclidiană ponderată după lungime între media segmentului $m_{\text{seg}}$ și media restului semnalului $m_{\text{out}}$ folosind sume prefixate (`cumsum`):
     $$\text{Score}(i, j) = \sqrt{\frac{1}{B}\sum_{b=1}^B (m_{\text{seg}, b} - m_{\text{out}, b})^2} \cdot \sqrt{j - i}$$
4. **Rafinarea Limitelor (`refine_boundaries`):**
   - Proiectează cadrele pe vectorul diferență $d = m_{\text{seg}} - m_{\text{out}}$ (filtrare adaptată / matched filtering).
   - Identifică tranzițiile de început și sfârșit prin maximizarea gradienților locali în treaptă ai curbei de proiecție.

## 3. Detalii de Implementare (`main.py`)
- **Încărcarea Fișierelor Audio (`read_wav`):** `scipy.io.wavfile` cu fallback pe modulul din biblioteca standard `wave`. Convertește semnalul stereo în mono și normalizează adâncimile de biți întregi (`uint8`, `int16`, `int24`, `int32`).
- **Parametri STFT:** Dimensiunea ferestrei $win \approx 0.093 \cdot f_s$ (putere a lui doi, minim 512 eșantioane) cu pasul de salt (hop size) $win // 8$. Masca de activitate filtrează tăcerea profundă ($< -20\,\text{dB}$ față de energia mediană).
- **Pipeline de Inferență:** Calculează $Z$, scanează segmente la intervale de 0.25s (0.4s pentru fișiere audio mai lungi), rafinează limitele la precizie de milisecunde și scrie `detection.csv`.

## 4. Rezultate și Artefacte
- **Validare:** Pe fișierul `train-input.wav`, limitele detectate au corespuns valorilor reale din `train-output.csv` la nivel de zeci de milisecunde (mult sub pragul permis de $\pm 1.5\,\text{s}$).
- **Stare Platformă:** Au fost înregistrate 15 încercări de trimitere (`teamStatus: "rejected"` în `meta.json`); detectorul criminalistic audio independent finalizat este conservat în `main.py`.
- **Fișiere:** `main.py`.
