# Raport de Soluție: task_angola („The Waterline Replay”)

## 1. Prezentare Generală și Obiectiv
- **ID Sarcină:** `task_angola` (Angola).
- **Dificultate și Scor:** Medie, 200 / 200 puncte.
- **Domeniu / Track:** Criminalistică pentru Agenți LLM (LLM Agent Forensics), Audit de Memorie RAG.
- **Context:** Un asistent IA de servicii municipale din Luanda, Angola, a halucinat un anunț eronat de oprire de urgență a apei pentru cartierul Kilamba.
- **Date de Intrare:** `test-input.json` conținând loguri structurate de trasare (trace logs) din execuția mai multor rulări ale unor agenți LLM autonomi. Vectorul `events` înregistrează:
  - Identificatorii rulării (`run_id`) și marcajele temporale;
  - Tipurile de evenimente (`event`);
  - Invocările de unelte și răspunsurile acestora;
  - Elemente de regăsire a contextului din memoria pe termen lung (`memory.search`);
  - Conținutul prompturilor și completările generate de asistent.
- **Obiectiv:** Auditarea logului de trasare al agentului pentru a izola intrarea coruptă din baza de cunoștințe care a declanșat halucinația, extrăgând parametrul exact `source_id`.
- **Format Flag Țintă:** Șir de caractere `SIGMOID_<PHRASE>`.

---

## 2. Perspective Cheie și Metodologie

Erorile din fluxurile RAG provin din distorsiunea interogărilor, regăsirea de memorie coruptă sau eșecuri de sinteză. În această provocare criminalistică, elementul de cunoștințe corupt este etichetat direct în evenimentele de trasare ale nivelului de regăsire din memoria pe termen lung a agentului (`memory.search`):
1. **Inspectarea Automată a Fluxului de Evenimente:** Parsarea fișierului `test-input.json` și iterarea prin vectorul `events`.
2. **Filtrarea Operațiunilor de Regăsire:** Detectarea evenimentelor unde `ev.get("event") == "memory.search"`.
3. **Extragerea Identificatorului:** Inspectarea listei `result_ids` din cadrul evenimentelor potrivite pentru a identifica șirurile ce poartă prefixul de flag al platformei `SIGMOID_`.

---

## 3. Detalii de Implementare Pas cu Pas

### Script de Implementare (`solve.py`)
Scriptul realizează parsarea deterministă și trimiterea către platformă:

```python
import json
from pathlib import Path
from optimizations.sigmoid_api import SigmoidClient


def main():
    path = Path("challenges/task_angola/data/test-input.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    flag = None
    for ev in data["events"]:
        if ev.get("event") == "memory.search" and "SIGMOID_" in str(ev.get("result_ids")):
            for rid in ev.get("result_ids", []):
                if rid.startswith("SIGMOID_"):
                    flag = rid
                    break

    print(f"Identified Faulty Memory ID: {flag}")
    if flag:
        client = SigmoidClient()
        res = client.submit_text("task_angola", flag)
        print("Submission result:", res)
    return flag


if __name__ == "__main__":
    main()
```

---

## 4. Rezultate și Artefacte
- **Stare:** `PASSED` (Scor: **200 / 200** puncte).
- **Tentativă de Trimitere:** Tentativa 1 (`2026-11-14T10:38:00.082Z`, trimis de: `Kozak`, 0 tentative incorecte).
- **Flag Extras:**
  ```text
  SIGMOID_MEM_KILAMBA
  ```
- **Artefacte Cheie:**
  - `solve.py` (859 octeți, script automat de audit al trasării și trimitere).
