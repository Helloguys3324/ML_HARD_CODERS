# task_egypt — "Always Valid, No Excuses" (Easy, 100 pts, file submission)

## Deliverable

- `work/solutions/task_egypt/main.py` — the file to submit.
- Reads `test-input.txt` (UTF-8, one OCR card per line) from cwd or from the
  script's own directory, writes `test-output.txt` beside the input.
- One strict JSON object per input line, same order; nothing else on stdout.
- Pure Python 3.10–3.12, stdlib only. **No network, no model download needed at
  evaluation time.** Runtime on 100 records: ~0.45 s (well under the 60 s limit).
- `answer.txt` — submission pointer (file task; flag only released by evaluator).
- `validate.py` — reproducible validator (sample + spec example + edge cases).
- `proto.py` — first-pass prototype used to confirm the rule engine on the sample.
- `data/LMML_Egypt_TRAIN_DATA/` — extracted sample (`train-input.txt`,
  `train-output.txt`).

## Task recap (from `descriptionEn`)

Each input line is `key: value` fields joined by `|`, with (1) messy spacing and
(2) scan noise that is not a field. Rules:

1. Trim each key and each value.
2. Drop noise: page numbers (`page 22`), headers (`Chapter IX`), footers
   (`- 45 -`), bracketed marks (`[torn]`, `[illegible]`, `(margin)`), figure
   labels, garble (`###`). Only a segment shaped like `key: value` with a
   **simple word key** is a real field.
3. Type each value: empty → `null`; exactly `true`/`false` → JSON booleans;
   whole number (`40`, `0`, `-3`) → JSON number; anything else → JSON string
   (`"` and `\` escaped). Only the **first** `:` splits key from value
   (`ratio: 3:1` → `"ratio":"3:1"`).
4. Emit exactly the real fields, with the same data; key order/whitespace are
   irrelevant because the grader compares parsed data.

## Approach (rules + the local oracle, oracle never trusted blindly)

The supplied Qwen3-1.7B-GGUF oracle is a generator, not a validator; the task
story itself lists its failure modes (quoted numbers, missing fields, mangled
backslashes, retained torn page numbers). So `main.py` implements the intended
"hold it to the format" pattern:

1. **Deterministic rule engine** (authoritative):
   - split line on `|`; strip each segment;
   - drop segments without `:` (all documented noise patterns have no colon);
   - split at the **first** `:`; strip key/value;
   - keep only keys matching `^[A-Za-z]+$` ("simple word"): this also rejects
     colons hidden in headers/footers such as `Chapter IX: ...`, `page 22: ...`,
     `[torn]: ...`;
   - type the value with the exact spec rules. Integer regex is `^-?[0-9]+$`
     (ASCII, optional minus); floats/`+5`/`1e3`/`007`-style edge cases stay
     strings unless they are plain whole numbers.
2. **Optional local oracle**: if `models/*.gguf` (Qwen3-1.7B-Q8_0.gguf) and
   `llama-cpp-python` are present, the model is consulted **only** for a
   non-blank record where the rule engine found zero real fields (never for
   normal cards). Its answer is re-parsed/normalised by the same rule engine
   (`_apply_hint`), so oracle drift cannot reach the output. If the model is
   absent (the reproducible case here — llama-cpp-python is not installed and
   the 1.8 GB GGUF is not bundled), the script runs offline with rules only.
   This keeps the 60 s GPU budget safe while honoring the "use the oracle"
   requirement.
3. **Strict-JSON guarantee**: every emitted line is
   `json.dumps(... ensure_ascii=True, allow_nan=False, separators=(",",":"))`
   and is round-tripped through `json.loads` with an assertion before writing.
   ASCII-safe escaping means a value containing U+2028/U+000B or any control
   character cannot split a record across physical lines.
4. **I/O robustness**: `utf-8-sig` read (BOM tolerant), universal newlines
   (CRLF/LF), drops exactly one trailing empty element from the final newline,
   writes exactly N lines with **no trailing newline** (correct under both
   `splitlines()` and plain `split("\n")`), explicit `open()` instead of
   `Path.read_text(newline=...)` (that kwarg only exists on Python ≥3.13, while
   the task runs 3.10–3.12).

## Evidence / validation

Commands and results (all run with the repo wrapper `/home/kozak/mllm/py`
unless a specific interpreter is stated):

```
$ /home/kozak/mllm/py validate.py
[sample] OK (13 lines)
[spec-example] OK (1 lines)
[edge] OK (6 lines)
ALL OK
```

- **Sample: 13/13 exact parsed-data matches** against `train-output.txt`
  (line-by-line `json.loads` equality), including:
  - `epithet : cost: $200` → `"cost: $200"` (first-colon rule),
  - `role: architect, vizier: and healer` → full string,
  - `title: "lord of the moon"` → `"\"lord of the moon\""`,
  - `scroll: C:\papyri\thoth` → `"C:\\papyri\\thoth"`,
  - empty values → `null`, `true`/`false` → booleans, `-18`/`0`/`-1264` → numbers,
  - every noise token from the sample (`page …`, `Chapter …`, `- 12 -`,
    `[torn]`, `[illegible]`, `[stain]`, `(margin)`, `(faded)`, `Fig 3`,
    `Plate VII`, `footnote 3`, `###`, `~~~`) dropped.
- **Spec example** from the description reproduced exactly:
  `page 22| king :Menes| title: ruled with "the Aten" |[torn]| alive :true | Chapter IX`
  → `{"king":"Menes","title":"ruled with \"the Aten\"","alive":true}`.
- **Edge cases** (typing/escaping/first-colon/noise-with-colons/blank line):
  all lines valid strict JSON, line count equals input count.
- **Fuzz/performance**: 100 synthetic records using the observed field
  vocabulary, tricky values and the full observed noise pool → 100/100 match an
  independently written reference implementation; 0.44 s total under
  `/nix/store/...-python3-3.12.8/bin/python3.12`; output is pure ASCII.
- **Grader-style run** under Python 3.12 (the task's target range): rc=0,
  13 output lines, exact parsed-data match, no trailing newline.

## Why the answer is right

The sample cards exercise every conversion rule stated in the task: trimming,
first-colon splitting, string escaping of `"` and `\`, empty→null, bool, int,
noise dropping, and garbage disposal. The rule engine reproduces all 13 expected
JSON objects exactly. The hidden set "uses different values: match the format,
not the sample data" — the format is fully covered by the implemented rules, and
the fuzz run over the same schema/noise families is 100/100. Output lines are
validated as strict JSON at write time, so the "one malformed card fails" risk
is eliminated by construction.

Residual risk: an unusual hidden noise token that both contains a colon and has
a single-word key (e.g. `Fig: 3`) would be indistinguishable from a real field
under the spec's own shape rule; and plus-signed integers (`+5`) are kept as
strings to match the canonical `-?\d+` whole-number convention. Both are absent
from the sample and contradict the stated generation format.

## Submission

File task. Primary artifact: `work/solutions/task_egypt/main.py`. Note the
platform metadata (`task.json`) advertises `acceptedFileExtensions: "txt"` even
though the description says "Submit `main.py`", so a **byte-identical** copy
`work/solutions/task_egypt/main.py.txt` is provided as an upload hedge
(`md5 c6693a0ce13bfd3abd6304fa431495b8` for both; Python executes a `.txt` file
passed on the command line regardless of extension, verified under Python 3.12).
Upload `main.py` if the API accepts it, otherwise upload `main.py.txt` — the
bytes and behaviour are identical. Do **not** submit from this subagent; the
orchestrator runs the evaluator, and only then is the `SIGMOID_<WORD>` flag
released.

CONFIDENCE: high
