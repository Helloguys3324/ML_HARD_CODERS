# Португалия — The Douro Paradox (`task_portugal`)

| Параметр | Значение |
|---|---|
| **Страна / Регион** | **Португалия (PT)** |
| **Категория** | **Причинно-следственный анализ / Парадокс Симпсона** |
| **Сложность** | **Medium** |
| **Баллы** | **200 / 200** (100% максимальный результат) |
| **Флаг решения** | `SIGMOID_DISTANCE_FROM_PORTO_KM` |
| **Статус на платформе** | ✅ **PASSED** |

---

## 1. 📖 Постановка задачи
В долине реки Дору тестировали новый метод капельного орошения виноградников. В целом по выборке новый метод побеждал контрольный (30.95% против 27.04%). Однако стажер обнаружил переменную, разбиение по которой делает новый метод хуже старого во всех группах без исключения.

---

## 2. 🔍 В чем состоял подвох
Среди 14 признаков многие дают частичный реверс, но полный парадокс Симпсона во всех группах дает только географическое зонирование долины Дору.

---

## 3. ⚙️ Как работает решение (подтверждение строчками кода)
Ниже представлены ключевые строки исходного кода решения из файла `solve.py`, подтверждающие описанный алгоритм:

```python
import pandas as pd
from optimizations.sigmoid_api import SigmoidClient


def main():
    df = pd.read_csv("challenges/task_portugal/data/douro_vineyard_data.csv")

    # Split distance_from_porto_km into traditional Douro DOC subregions:
    # Baixo Corgo (<= 105km), Cima Corgo (105-160km), Douro Superior (> 160km)
    g1 = df[df["distance_from_porto_km"] <= 105]
    g2 = df[(df["distance_from_porto_km"] > 105) & (df["distance_from_porto_km"] <= 160)]
    g3 = df[df["distance_from_porto_km"] > 160]

    for i, g in enumerate([g1, g2, g3], 1):
        st = g[g["treatment"] == "treatment"]["harvest_success"].mean()
        sc = g[g["treatment"] == "control"]["harvest_success"].mean()
        print(f"Region {i}: Treatment={st:.4f}, Control={sc:.4f}, Diff={st-sc:+.4f}")
        assert st < sc, f"Region {i} does not exhibit reversal"

    flag = "SIGMOID_DISTANCE_FROM_PORTO_KM"
    print(f"Flag: {flag}")

    client = SigmoidClient()
    res = client.submit_text("task_portugal", flag)
    print("Submit result:", res)
```

---

## 4. 🏆 Результат и верификация
- **Итоговый флаг:** `SIGMOID_DISTANCE_FROM_PORTO_KM`
- **Результат проверки платформы:** Статус: passed, 200/200 баллов, Submission ID: 698.

---

## 5. 📂 Файлы решения в этой папке
- [`solve.py`](./solve.py)

---

## 🚀 Запуск решения
```bash
python solve.py
```
