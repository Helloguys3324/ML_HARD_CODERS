# Solution Report: task_finland ("Pipes, Leaks, and the Graph Beneath")

## 1. Problem Overview
- **Task & Track:** Medium (200 pts). The Finnish Maritime Ferry Authority compiled a two-year log of intercity routes (`routes.csv`) across 10 regional offices. Due to bilingual operations (Finnish/Swedish), inconsistent spelling, missing Scandinavian umlauts (`ä`, `ö`, `å`), and punctuation noise, naive frequency counts misidentified Orivesi as the primary transport hub.
- **Objectives:**
  1. Perform entity resolution and normalization on `from_town` and `to_town`.
  2. Reconstruct the undirected graph without multi-edges or self-loops.
  3. Satisfy exact graph invariants:
     - Exactly **116** unique towns (vertices).
     - Exactly **448** unique routes (undirected edges).
     - Strictly unique degrees among the top-8 busiest cities.
  4. Generate flag `SIGMOID_<WORD>`, where `<WORD>` is capitalized from the initial letters of the top-7 cities by degree (descending).

## 2. Key Insight & Methodology
Bilingual municipality naming created extensive multi-aliasing across Swedish and Finnish forms, coupled with stripped ASCII diacritics.

1. **Swedish-to-Finnish Translation (`SWEDISH_TO_FINNISH`):** Over 65 mappings covering native Scandinavian and ASCII-stripped forms (e.g., *Åbo/Abo* $\to$ *Turku*, *Helsingfors* $\to$ *Helsinki*, *Tammerfors* $\to$ *Tampere*, *Uleåborg/Uleaborg* $\to$ *Oulu*, *Tavastehus* $\to$ *Hämeenlinna*, *Borgå/Borga* $\to$ *Porvoo*, *Esbo* $\to$ *Espoo*).
2. **Diacritic Restoration (`DIACRITICS`):** 23 mappings restoring standard Finnish orthography (e.g., *Jarvenpaa* $\to$ *Järvenpää*, *Jyvaskyla* $\to$ *Jyväskylä*, *Hameenlinna* $\to$ *Hämeenlinna*).
3. **Punctuation & Whitespace Stripping:** Stripping `.!?,_-\t\r\n` followed by `.title()` casing.

## 3. Implementation Details (`solve.py`)
- **Cleaning Function (`clean_town`):**
  ```python
  def clean_town(name):
      name = name.strip(" .!?,_-\t\r\n").title()
      if name in SWEDISH_TO_FINNISH:
          name = SWEDISH_TO_FINNISH[name]
      if name in DIACRITICS:
          name = DIACRITICS[name]
      return name
  ```
- **Graph Invariant Checks:**
  - Evaluates unique vertices: `len(all_towns) == 116`.
  - Filters self-loops ($u = v$) and deduplicates undirected edges via `tuple(sorted([u, v]))`: `len(edges) == 448`.
- **Top Hub Extraction:**
  Computing vertex degrees `len(adj[t])` reveals the top-7 hubs:
  1. **H**elsinki
  2. **O**ulu
  3. **T**ampere
  4. **L**ahti
  5. **I**isalmi
  6. **N**aantali
  7. **E**spoo

## 4. Results & Artifacts
- **Graph Verification:**
  - Vertices: **116** (verified).
  - Edges: **448** (verified).
  - Acronym: **HOTLINE**.
- **Flag:** `SIGMOID_HOTLINE`.
- **Files:** `solve.py`.
