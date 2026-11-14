# Raport de Soluție: task_estonia („This Data Is Perfect!”)

## 1. Prezentare Generală a Problemei
- **Sarcină și Domeniu (Track):** Medie (Medium, 200 pct). Ministerul Afacerilor Digitale din Estonia a creat un model de detecție a fraudelor pentru e-Rezidenți (1 = fraudă, 0 = legitim) care pretinde o acuratețe de **93.42%** utilizând un clasificator 1-NN (`KNeighborsClassifier(n_neighbors=1)`) cu `StandardScaler` pe 6 caracteristici numerice din `applications.csv` (1.971 înregistrări).
- **Obiectivele Auditului:**
  1. Auditarea `project_pipeline.py` pentru a identifica trei decizii inginerești discutabile.
  2. Demonstrarea experimentală că două dintre ele sunt piste false (decoy-uri) care nu sunt responsabile pentru metrica umflată.
  3. Identificarea cauzei rădăcină reale (scurgerea de date / data leakage).
  4. Corectarea pipeline-ului de validare pentru a calcula o acuratețe nepărtinitoare prin Leave-One-Out / validare încrucișată pe grupuri (group cross-validation).
  5. Trimiterea flag-ului `SIGMOID_<ACCURACY>`, unde `<ACCURACY>` este procentul întreg real.

## 2. Analiza Cauzei Rădăcină și Ipoteze Cheie
1. **Pista Falsă 1 (Decoy 1): Scalarea caracteristicilor înainte de împărțire (Pre-Split Feature Scaling):**
   - În pipeline-ul original, `StandardScaler().fit_transform(X)` a fost executat pe întregul set de date înainte de `train_test_split`.
   - *Test:* Ajustarea `StandardScaler` strict pe fold-urile de antrenament în cadrul unui `Pipeline` a generat o acuratețe de **93.67%** (o diferență de doar +0.25%, adică exact 1 eșantion din 395). Scalarea pre-split nu a fost cauza umflării metricii.
2. **Pista Falsă 2 (Decoy 2): Împărțire aleatorie unică nestratificată (Single Unstratified Random Split):**
   - Codul inițial a evaluat un singur apel `train_test_split(..., test_size=0.2, random_state=1137)`.
   - *Test:* `LeaveOneOut()` standard la nivel de rând a produs o acuratețe de **95.94%** (split stratificat: 93.92%). Metrica a rămas artificial de ridicată, eliminând ipoteza unei erori de eșantionare aleatorie.
3. **Eroarea Reală: Scurgere de date la nivel de grup / identitate prin eliminarea identificatorilor (Group / Identity Leakage via Dropped Identifiers):**
   - Inginerii au eliminat coloanele de identificare:
     ```python
     X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
     ```
   - *Anatomia Scurgerii:* 1.971 de aplicații aparțineau unui număr de numai 900 de indivizi unici (`date_of_birth` + `citizenship`). 1.611 cereri au fost depuneri repetate (între 2 și 4 per individ, cu statut constant de fraudă). Caracteristicile cererilor repetate au fost aproape identice (distanța euclidiană medie intra-persoană a fost de 0.038 față de 3.365 între persoane distincte).
   - În setul de test aleatoriu, **78.0% dintre eșantioanele de test (308/395)** aveau înregistrări corespondente de la același solicitant în setul de antrenament. Clasificatorul 1-NN a obținut **100.0% (308/308)** pe solicitanții cu date scurse, dar numai **71.3% (62/87)** pe solicitanții cu adevărat nevăzuți anterior.
- **Soluție:** Validare conștientă de grupuri folosind `LeaveOneGroupOut`, grupată după `date_of_birth` + `citizenship`.

## 3. Detalii de Implementare (`solve.py`)
- **Caracteristici (6 numerice):** `form_fill_seconds`, `doc_scan_quality`, `photo_match_score`, `email_domain_age_days`, `ip_geo_distance_km`, `address_history_years`.
- **Grupare:** `groups = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)` (900 de grupuri unice).
- **Reproducerea Baseline-ului:** `StandardScaler` global + `train_test_split(test_size=0.2, random_state=1137)` + `KNeighborsClassifier(n_neighbors=1)` au reprodus valoarea pretinsă de 93.42%.
- **Evaluarea Onestă a Pipeline-ului:**
  ```python
  pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=1))])
  preds_logo = cross_val_predict(pipe, X, y, groups=groups, cv=LeaveOneGroupOut(), n_jobs=-1)
  true_acc = accuracy_score(y, preds_logo)
  ```

## 4. Rezultate și Artefacte
- **Predicții Corecte:** 1.511 / 1.971.
- **Acuratețe Reală Out-of-Group:** `0.766616` (76.66%), rotunjită la **77%**.
- **Flag:** `SIGMOID_77`.
- **Fișiere:** `solve.py`.
