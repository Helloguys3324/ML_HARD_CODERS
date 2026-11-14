#!/usr/bin/env python
"""Rebuild main.py from main_template.py using the vocabulary payload and
registry already embedded in an existing main.py.

This avoids re-downloading / re-serialising the 8.6 MB of tokenizer data while
allowing the solver logic to evolve independently.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def extract(src_path):
    src = open(src_path, encoding="utf-8").read()

    m = re.search(r'_DATA = r"""(.*?)"""', src, re.S)
    if not m:
        raise SystemExit("could not locate _DATA payload in %s" % src_path)
    data = m.group(1)

    m = re.search(r"^_REGISTRY = (\[.*\])\s*$", src, re.M)
    if not m:
        raise SystemExit("could not locate _REGISTRY in %s" % src_path)
    registry = m.group(1)
    return data, registry


def main():
    old_path = os.path.join(HERE, "main.py")
    template_path = os.path.join(HERE, "main_template.py")
    out_path = os.path.join(HERE, "main.py")

    data, registry = extract(old_path)
    template = open(template_path, encoding="utf-8").read()
    if "__VOCAB_DATA__" not in template or "__REGISTRY__" not in template:
        raise SystemExit("template placeholders missing")

    out = template.replace("__VOCAB_DATA__", data).replace("__REGISTRY__", registry)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)

    # byte-identical copy for platforms that insist on a .json upload
    with open(os.path.join(HERE, "main.json"), "w", encoding="utf-8") as fh:
        fh.write(out)

    print("wrote %s (%.2f MB)" % (out_path, os.path.getsize(out_path) / 1e6),
          file=sys.stderr)


if __name__ == "__main__":
    main()
