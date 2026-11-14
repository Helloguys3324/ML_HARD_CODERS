import json, re, sys

KEY_RE = re.compile(r"^[A-Za-z]+$")
INT_RE = re.compile(r"^[+-]?[0-9]+$")


def convert_value(v: str):
    if v == "":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if INT_RE.match(v):
        try:
            return int(v)
        except ValueError:
            pass
    return v


def parse_record(line: str):
    obj = {}
    for seg in line.split("|"):
        seg = seg.strip()
        if ":" not in seg:
            continue
        key, value = seg.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not KEY_RE.match(key):
            continue
        obj[key] = convert_value(value)
    return obj


def main():
    base = sys.argv[1]
    inp = open(base + "/train-input.txt", encoding="utf-8-sig").read().splitlines()
    exp = open(base + "/train-output.txt", encoding="utf-8-sig").read().splitlines()
    ok = 0
    for i, (line, e) in enumerate(zip(inp, exp), 1):
        got = parse_record(line)
        want = json.loads(e)
        same = got == want
        ok += same
        if not same:
            print(f"LINE {i} MISMATCH\n  got {json.dumps(got, ensure_ascii=False)}\n  exp {e}")
        else:
            # also verify strict one-line json roundtrip
            s = json.dumps(got, ensure_ascii=False, separators=(",", ":"))
            assert json.loads(s) == want
    print(f"{ok}/{len(exp)} match")


if __name__ == "__main__":
    main()
