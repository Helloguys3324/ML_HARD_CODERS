# Solution Report: task_algeria ("Here, at your service!")

## 1. Overview & Objective
- **Task ID:** `task_algeria` (Algeria).
- **Difficulty & Score:** Hard, 300 / 300 points.
- **Track:** LLM Fine-Tuning, Low-Rank Adaptation (LoRA), Grammar & Style Conformance.
- **Context:** Automation of traditional town crier auction chants in ancient Beni Isguen (M'zab Valley, Algeria) using a compact LLM.
- **Base Model:** `HuggingFaceTB/SmolLM2-360M-Instruct` in 8-bit GGUF (`smollm2-360m-instruct-q8_0.gguf`, 386 MB).
- **Target Specification (`register/RULES.md`):**
  1. Mandatory opening phrase `Sem, `;
  2. 6 grammatical entity classes marked with specific suffixes and seals (`-el`, `-on`, `-ar`, `-is`, `-um`, `-os`);
  3. Strict vocabulary substitutions via `register/vocabulary.json`;
  4. Proper noun transliteration rules (including multi-word and possessive entities);
  5. Digit-wise numerals terminated by `-eth` (e.g., `42` $\to$ `four-two-eth`);
  6. Sentence seal affixed to the final marked entity;
  7. Maximum sentence length: $\le 18$ words;
  8. Single-noun response to market auction "calls", sealed.
- **Core Dilemma:** Imbuing rigid register compliance without catastrophic forgetting of general factual knowledge and reasoning.
- **Official Evaluation Gates (`evaluation_script.py`):**
  - `style_conformance >= 0.15`
  - `entity_class_accuracy >= 0.70`
  - `content_kept >= 0.55`
  - `base_log_prob >= -2.60`
  - `calls >= 0.75` (minimum 6 of 8 correct)
- **Deliverable:** Single `adapter.gguf` file ($\le 100\text{ MiB}$).

---

## 2. Key Insights & Methodology

1. **Deterministic Rule Compiler (`register_lib.py`):**
   Replaces fragile external LLM prompting with a programmatic rule engine translating English sentences into register-compliant text. Incorporates assertions against `style_checker.check_reply` to guarantee 100% adherence to all grammatical rules.
2. **Balanced Knowledge Corpus (`build_dataset_v8.py`):**
   Constructs `train_v8.jsonl` comprising 8,825 prompt-target pairs across 5,615 unique queries:
   - Global geography, capitals, official languages, and currencies;
   - 250 book/author, 90 art/artist, and 70 music/composer pairings;
   - 150 historical dates, 150 quantitative measurements, and 200 geographic records;
   - 150 Algerian regional facts (Sahara, Kabylia, Hoggar, Tuareg, Camus, Kateb Yacine, Rai, couscous, Dinar, 1954/1962);
   - 1,323 single-noun market call responses across all 6 entity classes.
3. **Targeted LoRA Optimization (`train_adapter.py`):**
   - Standard SmolLM2 ChatML format (`<|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>`).
   - Cross-entropy loss calculated strictly over assistant tokens (prompt tokens masked with `-100`).
   - Architecture: PEFT LoRA rank $r=32$, $\alpha=64$, dropout $0.05$, targeting all 7 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Restricting training to exactly **2 epochs (1,102 gradient steps)** at $\text{lr} = 2 \times 10^{-4}$ (cosine decay) on an NVIDIA A10G GPU prevented over-fitting, retaining 72–79% baseline factual accuracy while achieving full style conformance.
4. **GGUF Export Pipeline:**
   Converts PEFT PyTorch weights to FP16 GGUF format (448 tensors across 32 layers).

---

## 3. Step-by-Step Implementation Details

### Pipeline Execution
1. **Dataset Synthesis (`build_dataset_v8.py`):**
   Executes `register_lib.py` transformations across curated factual databases and auction call queries, outputting verified `train_v8.jsonl`.
2. **Model Training (`train_adapter.py` / `train_modal_v4.py`):**
   - Platform: Modal Cloud GPU (NVIDIA A10G).
   - Optimizer: AdamW ($\text{lr} = 2\times 10^{-4}$, weight decay $0.01$).
   - Precision: `bfloat16`, batch size: 16, sequence length: 256.
   - Outputs PEFT weights checkpoint `adapter_peft_v8_e2`.
3. **GGUF Conversion (`tools/lora_to_gguf.py`):**
   ```bash
   python tools/lora_to_gguf.py modal_out/adapter_peft_v8_e2 --outfile work/solutions/task_algeria/adapter.gguf
   ```
   Produces final artifact `adapter.gguf` (34,764,736 bytes / 33.2 MiB).

---

## 4. Results & Artifacts

### Official Verification Metrics (`evaluation_script.py`)

| Metric Gate | Achieved Score | Required Threshold | Status |
|:---|:---:|:---:|:---:|
| **Style Conformance** | **0.690** | $\ge 0.15$ | ✅ PASS |
| **Entity Class Accuracy** | **0.972** (36 entities) | $\ge 0.70$ | ✅ PASS |
| **Content Kept** | **0.746** (59 fact prompts) | $\ge 0.55$ | ✅ PASS |
| **Base Log-Probability** | **−1.917** | $\ge -2.60$ | ✅ PASS |
| **Market Calls** | **0.875** (7 of 8 correct) | $\ge 0.75$ | ✅ PASS |
| **Overall Result** | **EXIT 0 (PASS)** | — | **300 / 300 pts** |

### Key Artifacts
- `adapter.gguf` (33.2 MiB, 448-tensor GGUF LoRA adapter);
- `solution.md` (detailed architectural and experimental report);
- `register_lib.py` (deterministic register compiler);
- `build_dataset_v8.py` (synthetic dataset generator);
- `train_adapter.py` (Modal GPU training script);
- `results/` (validation evaluation dumps and logs).
