# Кот-д'Ивуар — The Abidjan Transit Ledger (`task_cote_divoire`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Кот-д'Ивуар (CI)** |
| **Категория** | **Машинное обучение / Непрерывное обучение (Continual Learning)** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_REHEARSAL` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
Роутер SQL-запросов после дообучения на новых пространственных запросах полностью забыл старые транзакционные шаблоны (точность на старой задаче упала до 0.00).

---

## 2. 🔍 В чем состоял подвох
Это не бэкдор и не скрытый класс, а классическое катастрофическое забывание (catastrophic forgetting), вызванное полной перезаписью меток.

---

## 3. ⚙️ Как работает решение (подтверждение кодом)
1. Сформировали буфер повторения (rehearsal buffer), объединив 250 примеров новой задачи с 40 примерами старой задачи.
2. Построили классификатор на базе `FeatureUnion` (word n-grams 1-2 + char_wb 3-5) и `LinearSVC`.
3. Упаковали веса и данные в самодостаточный скрипт `main.py`.
4. Результат на закрытом тесте: **old-task accuracy 0.9733** (порог 0.85) и **new-task accuracy 0.9867** (порог 0.85).

### Ключевой фрагмент кода решения:
```python
# Из build_main.py: преодоление катастрофического забывания через rehearsal-буфер
rehearsal_data = new_spatial_samples + old_transactional_samples
model = Pipeline([
    ("features", FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2))), ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb"))])),
    ("clf", LinearSVC(C=1.0))
])
# Old task accuracy: 97.33%, New task accuracy: 98.67% -> SIGMOID_REHEARSAL
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_REHEARSAL`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, оба гейта пройдены.

---

## 5. 📂 Файлы решения в этой папке
- [`build_main.py`](./build_main.py)
- [`embedded_data.json`](./embedded_data.json)
- [`eval_cv.py`](./eval_cv.py)
- [`main.py`](./main.py)
- [`solve.py`](./solve.py)
- [`writeup.md`](./writeup.md)

---

## 🚀 Запуск решения
```bash
python main.py
```
