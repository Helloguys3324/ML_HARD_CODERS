# Raport de Soluție: task_rwanda ("The Drying Floor at Karongi")

## 1. Prezentare Generală și Obiectiv
- **Task ID:** `task_rwanda` (Rwanda).
- **Dificultate și Scor:** Easy, 100 / 100 puncte.
- **Domeniu:** General AI / Automate de Stare pentru Agenți (LangGraph).
- **Format Ieșire:** Corectarea erorii în `graph.py`, deblocând flag-ul `SIGMOID_<NAME>`.
- **Premisă:**
  - O cooperativă agricolă de cafea din Karongi (Lacul Kivu) a automatizat monitorizarea ariei de uscare printr-un automat de stare bazat pe LangGraph:
    $$\text{receive\_batch} \longrightarrow \text{inspect\_batch} \longrightarrow (\text{gata?}) \longrightarrow \begin{cases} \text{dry\_batch} \to \text{inspect\_batch} \to \dots & (\text{neaprobat}) \\ \text{grade\_batch} \to \text{END} & (\text{aprobat}) \end{cases}$$
  - În codul inițial, fluxul de lucru era prins într-o buclă infinită. Loturile de cafea primite, chiar și cele din limitele de umiditate reglementate, se roteau la nesfârșit între inspecție și uscare până la atingerea adâncimii maxime de recursivitate (`GraphRecursionError`).
  - Obiectivul a fost diagnosticarea și remedierea bug-ului din definiția grafului de stare.

---

## 2. Perspective Cheie și Analiza Cauzei Rădăcină
- **Mecanismul Stării Grafului:**
  - Fluxul se execută peste un `StateGraph(BatchState)` unde atributele de stare sunt gestionate printr-un `TypedDict`.
  - Rutarea condiționată după măsurare este guvernată de:
    ```python
    def route_after_inspection(state: BatchState) -> Literal["dry_batch", "grade_batch"]:
        if state["approved"]:
            return "grade_batch"
        return "dry_batch"
    ```
- **Identificarea Cauzei Rădăcină (`graph.py`):**
  - În definiția inițială a grafului, muchia care pleca din `inspect_batch` a fost configurată ca o muchie directă necondiționată:
    ```python
    # Conectare Defectuoasă a Muchiei:
    builder.add_edge("inspect_batch", "dry_batch")
    ```
  - Deoarece `builder.add_conditional_edges(...)` nu a fost atașat nodului `inspect_batch`, routerul `route_after_inspection` era complet ocolit în execuție.
  - Fiecare lot era trimis necondiționat înapoi la `dry_batch`, indiferent de starea sa de aprobare sau de umiditatea măsurată.

---

## 3. Detalii de Implementare
Remedierea a fost aplicată direct în `graph.py`:

1. **Înlocuirea Muchiei Defectuoase:**
   - A fost eliminată muchia necondiționată hardcodată `builder.add_edge("inspect_batch", "dry_batch")`.
   - A fost înlocuită cu dispecerizarea condiționată intenționată:
     ```python
     builder.add_conditional_edges(
         "inspect_batch",
         route_after_inspection,
         {
             "dry_batch": "dry_batch",
             "grade_batch": "grade_batch",
         },
     )
     ```
2. **Garantarea Fluxului Terminal:**
   - S-a asigurat că `grade_batch` se termină curat în nodul `END`:
     ```python
     builder.add_edge("grade_batch", END)
     ```
3. **Invarianții Limitelor de Umiditate:**
   - S-a validat că `inspect_batch` verifică corect intervalul admis de umiditate:
     ```python
     def inspect_batch(state: BatchState) -> dict:
         ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
         return {"inspections": state["inspections"] + 1, "approved": ready}
     ```
4. **Verificarea Execuției:**
   - S-a verificat că loturile cu umiditate inițială conformă (de ex. 12.0%) sunt rutate direct către `grade_batch` și se termină la `END` dintr-o singură trecere, fără recurență.
   - Loturile neconforme iterează prin `dry_batch` până când intră în intervalul $[10.5\%, 12.5\%]$, terminând în mod fiabil.

---

## 4. Rezultate și Artefacte
- **Status:** PASSED
- **Scor:** 100 / 100 puncte
- **Flag Recuperat:** `SIGMOID_CONVERGENCE`
- **Fișier Sursă Modificat:** `graph.py`
