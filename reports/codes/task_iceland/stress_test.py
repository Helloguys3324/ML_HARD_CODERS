#!/usr/bin/env python3
"""Stress test for main.py: build a much messier synthetic export from a known
clean table, run the cleaner, and require exact recovery."""
import csv
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "stress_run")

STATIONS = [
    "Geysir", "Mývatn", "Hveragerði", "Hveravellir", "Landmannalaugar",
    "Þingvellir", "Krýsuvík", "Eyjafjallajökull", "Askja", "Lýsuhóll",
    "Seyðisfjörður", "Akureyri", "Reykjanes", "Snæfellsnes", "Jökulsárlón",
    "Húsavík", "Vík", "Borgarnes", "Ísafjörður", "Egilsstaðir",
]


def mojibake(name, depth):
    """Simulate UTF-8 bytes decoded as latin-1, `depth` times."""
    s = name
    for _ in range(depth):
        s = s.encode("utf-8").decode("latin-1")
    return s


def fmt_temp(value, decimal_comma):
    text = f"{value:.1f}"
    return text.replace(".", ",") if decimal_comma else text


def main():
    rng = random.Random(20261114)
    os.makedirs(WORK, exist_ok=True)

    n = 120
    base = []
    used = set()
    for _ in range(n):
        rid = rng.randint(1, 400)
        while rid in used:
            rid = rng.randint(1, 400)
        used.add(rid)
        name = rng.choice(STATIONS)
        temp = round(rng.uniform(35.0, 99.0), 1)
        date = f"2026-{rng.randint(5, 9):02d}-{rng.randint(1, 28):02d}"
        base.append((rid, name, temp, date))
    base.sort()

    raw = []
    for rid, name, temp, date in base:
        copies = rng.choice([1, 1, 2, 2, 3, 4])
        for _ in range(copies):
            use_f = rng.random() < 0.55
            comma = rng.random() < 0.5
            depth = rng.choices([0, 1, 2, 3], weights=[5, 4, 2, 1])[0]
            broken = mojibake(name, depth)
            if use_f:
                value = round(temp * 9.0 / 5.0 + 32.0, 1)
                unit = "F"
            else:
                value = temp
                unit = "C"
            raw.append(f"{rid};{broken};{fmt_temp(value, comma)};{unit};{date}")

    rng.shuffle(raw)
    preamble = []
    for i in range(rng.randint(10, 40)):
        if i % 4 == 3:
            preamble.append("")
        elif i % 5 == 0:
            preamble.append("# reading_id;station_name;temperature_raw;unit;reading_date")
        else:
            preamble.append(f"# legacy relay block {i}: do not edit, encoding=ISO-8859-1")

    inp = os.path.join(WORK, "test-input.csv")
    with open(inp, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(preamble) + "\r\n")
        fh.write("reading_id;station_name;temperature_raw;unit;reading_date\r\n")
        fh.write("\r\n".join(raw) + "\r\n")

    main_py = os.path.join(HERE, "main.py")
    proc = subprocess.run(
        [sys.executable, main_py], cwd=WORK, capture_output=True, text=True
    )
    print("cleaner exit:", proc.returncode)
    if proc.stdout.strip():
        print("stdout:", proc.stdout.strip())
    if proc.stderr.strip():
        print("stderr:", proc.stderr.strip())
    assert proc.returncode == 0, "cleaner crashed"

    out_path = os.path.join(WORK, "test-output.csv")
    with open(out_path, encoding="utf-8", newline="") as fh:
        got = list(csv.reader(fh))

    expected = [["reading_id", "station_name", "temperature_c", "reading_date"]]
    for rid, name, temp, date in base:
        expected.append([str(rid), name, f"{temp:.1f}", date])

    assert got[0] == expected[0], f"header mismatch: {got[0]}"
    assert len(got) == len(expected), f"row count {len(got)} != {len(expected)}"
    errors = 0
    for g, e in zip(got[1:], expected[1:]):
        if g != e:
            errors += 1
            if errors <= 15:
                print("MISMATCH got", g, "expected", e)
    print(f"rows: {len(expected)-1}, mismatches: {errors}")
    assert errors == 0, "stress test failed"

    # Raw byte check of formatting conventions.
    with open(out_path, "rb") as fh:
        data = fh.read()
    assert b"\r\n" in data
    assert b'"' not in data
    print("STRESS TEST PASSED")


if __name__ == "__main__":
    main()
