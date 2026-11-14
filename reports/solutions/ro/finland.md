# Raport de Soluție: task_finland („Pipes, Leaks, and the Graph Beneath”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Medie (Medium, 200 pct). Autoritatea Maritimă de Feriboturi din Finlanda a compilat un istoric pe doi ani al rutelor interurbane (`routes.csv`) din 10 birouri regionale. Din cauza operațiunilor bilingve (finlandeză/suedeză), a ortografiei inconsecvente, a lipsei caracterelor speciale scandinave (`ä`, `ö`, `å`) și a zgomotului de punctuație, numărătoarea naivă a frecvențelor a indicat eronat orașul Orivesi drept nod principal de transport.
- **Obiective:**
  1. Efectuarea rezoluției și normalizării entităților (entity resolution) pentru `from_town` și `to_town`.
  2. Reconstrucția grafului neorientat fără muchii multiple sau bucle (self-loops).
  3. Respectarea invariantelor exacte ale grafului:
     - Exact **116** orașe unice (vârfuri).
     - Exact **448** rute unice (muchii neorientate).
     - Grade strict unice pentru primele 8 cele mai aglomerate orașe.
  4. Generarea flag-ului `SIGMOID_<WORD>`, unde `<WORD>` este format cu majuscule din literele inițiale ale primelor 7 orașe ordonate după grad (descrescător).

## 2. Perspectivă Cheie și Metodologie
Denumirile bilingve ale municipalităților au creat multiple pseudonime în limbile suedeză și finlandeză, combinate cu eliminarea semnelor diacritice în codificarea ASCII.

1. **Traducere din Suedeză în Finlandeză (`SWEDISH_TO_FINNISH`):** Peste 65 de mapări care acoperă formele scandinave native și variantele ASCII simplificate (de ex., *Åbo/Abo* $\to$ *Turku*, *Helsingfors* $\to$ *Helsinki*, *Tammerfors* $\to$ *Tampere*, *Uleåborg/Uleaborg* $\to$ *Oulu*, *Tavastehus* $\to$ *Hämeenlinna*, *Borgå/Borga* $\to$ *Porvoo*, *Esbo* $\to$ *Espoo*).
2. **Restaurarea Semnelor Diacritice (`DIACRITICS`):** 23 de mapări pentru restabilirea ortografiei standard finlandeze (de ex., *Jarvenpaa* $\to$ *Järvenpää*, *Jyvaskyla* $\to$ *Jyväskylä*, *Hameenlinna* $\to$ *Hämeenlinna*).
3. **Eliminarea Semnelor de Punctuație și a Spațiilor:** Eliminarea caracterelor `.!?,_-\t\r\n` urmată de aplicarea formatării cu majusculă `.title()`.

## 3. Detalii de Implementare (`solve.py`)
- **Funcția de Curățare (`clean_town`):**
  ```python
  def clean_town(name):
      name = name.strip(" .!?,_-\t\r\n").title()
      if name in SWEDISH_TO_FINNISH:
          name = SWEDISH_TO_FINNISH[name]
      if name in DIACRITICS:
          name = DIACRITICS[name]
      return name
  ```
- **Verificarea Invariantelor Grafului:**
  - Evaluarea numărului de vârfuri unice: `len(all_towns) == 116`.
  - Filtrarea buclelor ($u = v$) și deduplicarea muchiilor neorientate prin `tuple(sorted([u, v]))`: `len(edges) == 448`.
- **Extragerea Principalelor Noduri de Rețea:**
  Calcularea gradelor vârfurilor `len(adj[t])` identifică primele 7 noduri urbane:
  1. **H**elsinki
  2. **O**ulu
  3. **T**ampere
  4. **L**ahti
  5. **I**isalmi
  6. **N**aantali
  7. **E**spoo

## 4. Rezultate și Artefacte
- **Verificarea Grafului:**
  - Vârfuri: **116** (verificat).
  - Muchii: **448** (verificat).
  - Acronim: **HOTLINE**.
- **Flag:** `SIGMOID_HOTLINE`.
- **Fișiere:** `solve.py`.
