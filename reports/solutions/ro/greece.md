# task_greece — „Just a siblings' thing” (GR, Medie, 200 pct)

## Răspuns

```
SIGMOID_ARES
```

**ÎNCREDERE: ridicată**

---

## 1. Ce cere sarcina

Trei clasificatori PyTorch de tip „oracol” — **Ares**, **Athena**, **Apollo** — trebuie
evaluați pe `test_dataset.csv` în raport cu `test_labels.csv` la un prag decizional
fix de **0.50**. Toți trei au **aceeași acuratețe de testare (accuracy)**, prin urmare câștigătorul
trebuie ales pe baza metricii care corespunde profilului de cost declarat pentru implementare:

> „missing a true catastrophic threat is far more expensive than raising a false
> alarm.” („ratarea unei amenințări catastrofale reale este mult mai costisitoare decât declanșarea unei alarme false.”)

Aceasta este o setare sensibilă la costuri cu **C_FN ≫ C_FP**, adică maximizarea
**sensibilității / ratei de recunoaștere (recall / sensitivity)** (echivalent cu minimizarea fals negativelor; reflectat de asemenea de F-beta cu β>1, acuratețea echilibrată, MCC, ROC-AUC și PR-AUC). Flag-ul este
`SIGMOID_<UPPERCASE MODEL NAME>` corespunzător modelului selectat.

## 2. Date și puncte de control (inspecție statică)

Set de date: 1.000 de întâlniri (encounters), **100 pozitive** (`is_catastrophic_threat=1`) și
**900 negative**; fără valori nule sau duplicate; ordinea rândurilor din `test_dataset.csv` și
`test_labels.csv` a fost verificată și este identică.

Fișierele `.pth` sunt arhive zip PyTorch de 1.9–2.0 KB care conțin doar
`linear.weight` (9 float32 = 36 B) și `linear.bias` (1 float32 = 4 B); încărcate
cu `torch.load(..., weights_only=True)` (fără execuție arbitrară de cod). SHA-256:

| fișier | sha256 |
|---|---|
| `model_ares.pth` | `bae94c0c496596c2add6cea248eff8f488585cd7d9298f16c6668fa871267ea5` |
| `model_athena.pth` | `48572235b26f7e1b9d996bfe027c6a64be7ac63bd8bfb3be590c6a593d05accd` |
| `model_apollo.pth` | `0807bcb6fed1c832fcac97e27b7c6415e2c113594bad0f8bc02063c32725a527` |

Toate cele trei puncte de control sunt detectoare de semn pe o singură caracteristică (pondere exact 1 pe o
singură caracteristică, bias 0), astfel încât `sigmoid(logit) ≥ 0.50 ⇔ feature ≥ 0`:

| model | ponderi | caracteristică utilizată |
|---|---|---|
| ARES | `[0,1,0,0,0,0,0,0,0]`, bias 0 | `oracle_omen` |
| ATHENA | `[0,0,0,0,1,0,0,0,0]`, bias 0 | `aegis_resonance` |
| APOLLO | `[0,0,0,0,0,0,0,1,0]`, bias 0 | `styx_disturbance` |

Nu au fost găsite date utile (payload) sau metadate codificate în fișierele de checkpoint în afara acestor tensori.

## 3. Metrici la pragul de 0.50 (1000 de rânduri, 100 pozitive)

| metrică | ARES | ATHENA | APOLLO |
|---|---:|---:|---:|
| acuratețe (accuracy) | **0.9500** | **0.9500** | **0.9500** |
| precizie (precision) | 0.6786 | 0.7500 | **0.9167** |
| **sensibilitate (recall / sensitivity)** | **0.9500** | 0.7500 | 0.5500 |
| specificitate (specificity) | 0.9500 | 0.9722 | **0.9944** |
| F1 | **0.7917** | 0.7500 | 0.6875 |
| F2 (β=2) | **0.8796** | 0.7500 | 0.5978 |
| acuratețe echilibrată (balanced accuracy) | **0.9500** | 0.8611 | 0.7722 |
| MCC | **0.7781** | 0.7222 | 0.6878 |
| ROC-AUC | **0.9471** | 0.8766 | 0.7859 |
| PR-AUC | **0.6970** | 0.6389 | 0.6105 |
| TP / FN | 95 / **5** | 75 / 25 | 55 / 45 |
| FP / TN | 45 / 855 | 25 / 875 | **5 / 895** |
| pozitive prezise | 140 | 100 | 60 |

Matrice de confuzie (rânduri = valori reale 0/1, coloane = valori prezise 0/1):

```
ARES   [[855,  45],     ATHENA [[875,  25],     APOLLO [[895,   5],
        [  5,  95]]             [ 25,  75]]             [ 45,  55]]
```

Toate cele trei acurateți sunt **exact 0.9500** — premisa scenariului este verificată
numeric în mod cert.

## 4. De ce ARES — regula de decizie operațională (business)

Seturile de predicții sunt strict **imbricate** (`apollo ⊂ athena ⊂ ares`), ceea ce reprezintă
curba clasică de compromis precision/recall la o acuratețe fixă:

```
(ares, athena, apollo) -> număr (neg, pos)
(0,0,0) -> 855 neg,   5 pos     # ratate de toate trei
(1,0,0) ->  20 neg,  20 pos     # prinse doar de Ares
(1,1,0) ->  20 neg,  20 pos     # prinse de Ares+Athena
(1,1,1) ->   5 neg,  55 pos     # prinse de toate trei
```

Model de cost `cost = FP + k·FN` cu `k = C_FN / C_FP`:

| k | ARES | ATHENA | APOLLO | câștigător |
|---:|---:|---:|---:|---|
| 0.5 | 47.5 | 37.5 | 27.5 | APOLLO |
| **1.0** | 50.0 | 50.0 | 50.0 | egalitate (= aceeași acuratețe) |
| 2.0 | **55.0** | 75.0 | 95.0 | **ARES** |
| 5.0 | **70.0** | 150.0 | 230.0 | **ARES** |
| 10.0 | **95.0** | 275.0 | 455.0 | **ARES** |

Pentru **orice** `k > 1` (orice context în care o amenințare catastrofală ratată costă mai mult
decât o alarmă falsă — iar cerința subliniază „far more expensive”), **ARES este oracolul
optim**. La `k = 1` toate trei sunt la egalitate, motiv pentru care acuratețea lor
este identică. Singurul avantaj al modelului APOLLO este precizia (puține alarme false), pe care
contextul operațional o consideră în mod explicit drept eroarea mai puțin costisitoare.

Verificare de consistență: ARES câștigă de asemenea la **orice altă metrică orientată pe recall** —
recall, F1, F2, balanced accuracy, MCC, ROC-AUC, PR-AUC — așadar concluzia nu depinde
de definiția unei singure metrici. Athena depășește Apollo la toate acestea; doar
precizia și specificitatea le ierarhizează în sens invers.

## 5. Reproducere

```bash
cd /home/kozak/mllm/MML/work/solutions/task_greece
/home/kozak/mllm/py solve.py          # afișează toate metricile + analiza decizională
```

Artefacte: `solve.py`, `solve_output.txt` (rulare capturată), `metrics.csv`,
`metrics.json`, `answer.txt`. Acest script doar deduce răspunsul; **nu a fost realizată nicio
trimitere**.

## 6. Note și diligență tehnică

- Pragul este aplicat funcției `sigmoid(logit)` exact la **0.50**, conform cerințelor; nicio
  probabilitate de test nu este exact 0.5, nu există valori NaN/inf.
- Punctele de control au fost analizate static (listare zip + `weights_only=True`), eliminând riscul
  execuției de cod prin pickle.
- Nu au fost găsite injectări de instrucțiuni (prompt-injection) sau comenzi ascunse în coloanele CSV,
  identificatori (`MYTH-0000..0999`, contigui), regiuni sau fazele lunii.
- Singurul răspuns alternativ plauzibil ar fi `SIGMOID_APOLLO` într-o politică
  **orientată exclusiv pe precizie** (cele mai puține alarme false). Cerința specifică explicit
  asimetria inversă de costuri, astfel încât această alternativă este respinsă.
