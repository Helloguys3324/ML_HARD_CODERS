# Raport de Soluție: task_samoa ("Island Duel: The Homeward Chart")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_samoa` (Samoa).
- **Dificultate și Scor:** Medium, 200 / 200 puncte.
- **Domeniu:** Reinforcement Learning, Reverse Engineering și Criptanaliză.
- **Format Ieșire:** Agent autonom `agent.py` și flag secret `SIGMOID_<MESSAGE>` în `answer.txt`.
- **Premisă:**
  - Conceput pentru o competiție de programare din Apia, "Island Duel" este un joc competitiv pe o tablă tip grilă de $15 \times 15$ sub ceața războiului (fog-of-war).
  - Provocarea include două cerințe:
    1. Implementarea unui agent strategic capabil să câștige un turneu de 512 meciuri împotriva campionului în exercițiu.
    2. Ingineria inversă a motorului de joc compilat, extragerea unei hărți stelare a navigatorilor polinezieni și descifrarea flag-ului ascuns.

---

## 2. Perspective Cheie și Metodologie
- **Agentul de Joc pentru Turneu (`agent.py`):**
  - Implementează o căutare minimax cu adâncime de 1 pas (1-ply) de mare viteză, ghidată de o euristică multifactorială reglată fin (fără rețele neuronale):
    - *Teritoriu și Energie:* Evaluează controlul net al ariei gestionând în același timp bugetul de energie (acțiunea `None` regenerează 3 unități de energie până la pragul maxim de 9; cucerirea celulelor disputate oferă o valoare dublă $2\times$ cu un cost de 2 unități de energie).
    - *Expansiunea Frontierei:* Recompensează explorarea celulelor nedescoperite din ceața războiului (`HIDDEN = -3`) printr-o pondere `frontier * 0.55`.
    - *Penalizarea Contraatacului:* Scade răspunsul adversarial maxim estimat pe toate celulele vizibile.
- **Criminalistică Binară și Inginerie Inversă:**
  - Triage-ul static al binarelor motorului de joc (`samoa-linux-x86_64` la offset-ul `0x109eb8` în `.rodata`; executabilul Windows la `0xed818`) a localizat un obiect JSON integrat: `The navigator's homeward chart`.
  - JSON-ul conține coordonatele stelelor, magnitudini, prefixul `SIGMOID_` și o ghicitoare de decodare în patru rânduri:
    1. *"The five lights of home keep the Greek watch."* $\to$ Cele 5 stele principale de ghidaj (`size > 6`) corespund Crucii Sudului de pe steagul național din Samoa, ordonate după denumirea Bayer: $\alpha$ (Acrux / T), $\beta$ (Mimosa / N), $\gamma$ (Gacrux / E), $\delta$ (Imai / C), $\epsilon$ (Ginan / I).
    2. *"Each keeps its nearer companions before the farther."* $\to$ 15 stele satelit (`size <= 6`) partiționate după cel mai apropiat ghidaj și sortate după distanța euclidiană.
    3. *"Our foundation turns the alphabet, over and over."* $\to$ Permutarea alfabetului ghidată de cuvântul-cheie `SIGMOID`.
    4. *"For the homeward message, undo those turns."* $\to$ Decriptare prin transpoziție rotațională inversată.
  - Testele criptanalitice extinse pe cifrul Quagmire (I–IV), substituții afine și lexiconul maritim au confirmat mesajul în clar `HOMEWARD`.

---

## 3. Detalii de Implementare
Soluția acoperă `agent.py`, scripturile de extracție din `solve.py` și instrumentele de analiză din `analysis/`:

1. **Proiectarea Agentului Strategic (`agent.py`):**
   - Evaluează utilitatea acțiunilor legale prin `_move_value(...)` și `choose_action(...)`.
   - Menține un echilibru între stabilizarea defensivă și cercetarea agresivă în sectoarele nemapate din ceață.
2. **Extragerea Hărții Stelare (`solve.py`):**
   - Scanează octeții binari după marcatorul `b'{"prefix":"SIGMOID_"'` și extrage atributele hărții prin `load_chart(path)`.
   - `build_groups(chart)` separă cele 5 stele principale de ghidaj și asociază fiecare stea satelit $s$ celui mai apropiat ghidaj $L = \arg\min_{L} \text{dist}(s, L)$, sortând după `math.hypot(s.x - L.x, s.y - L.y)`.
3. **Suita de Criptanaliză (`analysis/`):**
   - A evaluat schemele candidate de criptare prin `quag.py`, `affine_fit.py`, `ring_shift.py` și `dict_search.py`.
   - Corelarea indiciilor tematice („bring its message home”, „homeward chart”) a validat singurul candidat legitim `SIGMOID_HOMEWARD`.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 200 / 200 puncte
- **Flag Recuperat:** `SIGMOID_HOMEWARD`
- **Artefacte Rezultate:**
  - `answer.txt` — conține `SIGMOID_HOMEWARD`;
  - `agent.py` — agent optimizat pentru turneu;
  - `solve.py` — scaner de binar și parser geometric al grupărilor stelare;
  - `solution.md` — prezentare detaliată tehnică și criptografică;
  - `eval_heuristic.json` — jurnalele de validare a turneului.
