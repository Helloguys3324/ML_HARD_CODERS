# Runbook 01 — NLP (North America)

> Template: `templates/01_nlp_fast.py` · Track key: `nlp`
> CPU verdict: **strong for TF-IDF/linear/hashed n-grams; impossible for transformer
> fine-tuning in 2 h.** Default to `--mode tfidf`, not `--mode transformer`.
> Modes: `tfidf` (default), `transformer`, `hybrid`, `retrieval` (needs
> `--corpus`), `ner` (token/tag columns via `--text-col/--target-col`).

---

## 1. Fastest credible baseline (5–10 min)

```bash
export PY=/home/kozak/mllm/py REPO=/home/kozak/mllm/MML
cd "$REPO"
$PY lmml_master.py new lvl_06 --track nlp
$PY lmml_master.py solve lvl_06 -- \
  --mode tfidf --text-col text --target-col target --id-col id \
  --metric auto --folds 5 --svd-dim 32 --no-use-lgb --no-pretrained
$PY lmml_master.py check lvl_06 --sample lvl_06/data/sample_submission.csv
```

`--no-use-lgb` keeps the two linear models (fold-wise TF-IDF + Rust hashed
n-grams) and skips the SVD/LightGBM stack. On ≤50k documents this is 2–4 min.
Then run the full stack without `--no-use-lgb` (5–10 min).

If the CSV has multiple text fields, concatenate them with a separator **inside
solve.py** or pre-build a `text` column; do not let the template silently use only
one field.

---

## 2. Best CPU model

1. **TF-IDF word(1,3) + char_wb(3,5), sublinear, min_df=2, `LogisticRegression`
   (liblinear/saga) or `Ridge` for regression.** This is the backbone; it is
   deterministic and hard to beat on lexical tasks.
2. **Rust-hashed n-grams (1–3) + LightGBM** on `[linear OOF logits, SVD(64), text
   meta]`. Adds non-linear interactions; template 01 already implements this.
3. **Rank/logit blend** of TF-IDF-linear and hash-LGB. Different error modes;
   usually +0.003–0.015 AUC/F1.
4. **Frozen sentence embeddings + linear probe** only if the task explicitly allows
   pretrained models. `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2` on CPU:
   ~5–15 min for 50k short texts, then logistic regression. No fine-tuning.

Do **not** start DeBERTa/ModernBERT on CPU. Even `deberta-v3-small`, 5 folds, 3
epochs at 256 tokens is a multi-hour job on 4 cores.

---

## 3. Feature engineering that pays on CPU

- Word 1–3 + char_wb 3–5 TF-IDF; `sublinear_tf=True`, `min_df=2`, `max_features`
  150k–300k. Char n-grams catch typos, morphology, code-switching, hashtags.
- Text meta: length, word count, unique-word ratio, uppercase/digit/punctuation
  ratios, `!`/`?` counts, URL/mention/hashtag counts, newline count, average word
  length, stopword ratio. Cheap and orthogonal to TF-IDF.
- Length bucket (quantile bins) as a categorical for LightGBM.
- Duplicate/near-duplicate flags: exact hash, first/last 64 chars, MinHash if needed.
- For long text: head+tail truncation (first 256 + last 256 tokens) beats middle-only.
- Class imbalance: `class_weight="balanced"`, then tune the threshold on OOF.

---

## 4. Validation

- `StratifiedKFold(n_splits=5, shuffle=True)` for i.i.d. text.
- `StratifiedGroupKFold` by `author_id`/`thread_id`/`source` when present; otherwise
  duplicate leakage inflates CV by several points.
- Fit vectorizers **inside each fold** (the template does). Fitting TF-IDF on all
  train+test then CV is transductive and can leak vocabulary, though it is usually
  mild; keep it fold-wise for honest scores.
- Run adversarial validation on the sparse matrix or on SVD(32): if AUC > 0.75,
  inspect domain/source/time shift. Reweight or drop drifting features.
- Report both micro and macro F1 for multiclass; optimize the competition metric,
  not accuracy.

---

## 5. Common pitfalls

| Pitfall | Symptom | Fix |
| :--- | :--- | :--- |
| Duplicate texts across folds | CV 0.99, LB 0.80 | Group split by exact/near duplicate |
| Vectorizer fit on test | Slightly optimistic CV | Fold-wise fit; template default |
| Transformer attempted on CPU | 1–3 h for one fold | TF-IDF first; frozen embeddings only if allowed |
| Metric mismatch | Good logloss, bad F1 | Threshold sweep on OOF; macro vs micro |
| Empty/NaN predictions on rare classes | `check` fails | Fallback prior; clip; ensure all classes |
| Over-long sequences | Tokenizer truncation loses label cue | Head+tail; sentence selection |
| Label leakage in text | Near-perfect CV | Strip IDs/URLs/label words; inspect top coefficients |

---

## 6. Snippets

### 6.1 Minimal TF-IDF + linear baseline (copy into `solve.py`)

```python
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tr = pl.read_csv("data/train.csv"); te = pl.read_csv("data/test.csv")
texts = tr["text"].fill_null("").to_list(); test_texts = te["text"].fill_null("").to_list()
y = tr["target"].to_numpy()
is_clf = len(np.unique(y)) <= 20
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); pred = np.zeros(len(te))
for f, (i_tr, i_va) in enumerate(skf.split(texts, y)):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=200_000, dtype=np.float32)
    Xtr = vec.fit_transform([texts[i] for i in i_tr])
    Xva = vec.transform([texts[i] for i in i_va])
    Xte = vec.transform(test_texts)
    m = LogisticRegression(C=2.0, max_iter=300, solver="liblinear") if is_clf else Ridge(alpha=1.0)
    m.fit(Xtr, y[i_tr])
    if is_clf:
        oof[i_va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / 5
    else:
        oof[i_va] = m.predict(Xva); pred += m.predict(Xte) / 5
print("OOF AUC", roc_auc_score(y, oof) if is_clf else "regression")
pl.DataFrame({"id": te["id"], "target": pred}).write_csv("submissions/submission.csv")
```

### 6.2 Rust hashed n-grams + LightGBM stack

```python
from optimizations.rust_fast_ext import fast_hash_ngrams_batch
X = fast_hash_ngrams_batch(texts, min_n=1, max_n=3, dim=1 << 17, analyzer="word_char")
Xte = fast_hash_ngrams_batch(test_texts, min_n=1, max_n=3, dim=1 << 17, analyzer="word_char")
# X is CSR float32; feed to LightGBM directly (it accepts sparse), or hstack
# with SVD(64) + text-meta before fitting.
```

### 6.3 Threshold / calibration

```python
from optimizations.math_tricks import oof_optimal_threshold, calibrate_oof_and_test
thr_info = oof_optimal_threshold(y, oof)      # args: (y_true, oof_prob); returns a dict
thr = thr_info["threshold_median"]
cal_oof, cal_test = calibrate_oof_and_test(oof, y, pred, method="platt")
```

---

## 7. Runtime on 4 cores (shared box)

| Operation | Time |
| :--- | :--- |
| TF-IDF + LogisticRegression, 20k docs | 1–3 min |
| TF-IDF + linear, 50k docs | 3–6 min |
| Full `--mode tfidf --use-lgb`, 50k docs | 6–12 min |
| Rust hash 20k docs → 131k features | ~2 s |
| SVD(64) on 50k×200k | 1–3 min |
| Frozen MiniLM/BGE + linear, 50k docs (if allowed) | 10–20 min |
| DeBERTa-v3-small fine-tune, 1 fold CPU | 45–120 min — **do not** |

## 8. Do not attempt without GPU

- `--mode transformer` on >2k samples, or any `--pretrained` model >base.
- 5-fold × 3-epoch transformer sweeps; LLRD/AWP/FGM on CPU.
- LLM fine-tuning, QLoRA, vLLM; external LLM APIs unless explicitly allowed.
- Training a transformer from scratch (template supports it; it is not competitive).
