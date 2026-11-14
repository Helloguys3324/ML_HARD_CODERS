#!/usr/bin/env python3
"""Solve task_australia (The Drunken Mario Adventures / Marathon Lab).

What the app is
---------------
Marathon Lab is a Nuitka one-file Pygame application.  The one-file payload
(``KA`` + zstd stream at the end of the executable, terminated by an 8-byte
little-endian payload length) contains a standalone distribution with
``marathon-lab.bin``, ``lab/proven.npy`` and the Pygame runtime.

The relevant game data (recovered by importing the compiled modules inside the
running app; see ``probe_runtime.py``) is:

  lab.level.COIN_SITES = [7, 12, 17, 25, ... 206]   (20 coins, all on row 11)
  lab.fitness._SIGN    = {(7, 11): 1, (12, 11): 0, ...}

``lab.fitness.reward`` pays +COIN_WEIGHT (25) for coins whose sign is 1 and
-COIN_WEIGHT for coins whose sign is 0.  A merely competent player would grab
every coin.  A fully trained agent learns to collect exactly the +1 coins and
to leave the -1 coins alone.

``lab/proven.npy`` is the bundled, fully-trained reference brain.  Running it
is deterministic: it reaches the flagpole in 1197 steps (fitness 514.21,
``won=True``) and takes exactly the seven +1 coins:
    7, 17, 80, 101, 110, 118, 172
and none of the thirteen -1 coins.  (Verified 3/3 identical runs.)

Reading the message
-------------------
Read the coin row left to right and write 1 for "trained agent wants it"
(``_SIGN`` == 1) and 0 for "trained agent avoids it" (``_SIGN`` == 0):

    1 0 1 0 0 0 0 0 0 1 0 1 1 1 0 0 1 0 0 0
    = 10100 00001 01110 01000

Split into four 5-bit groups and decode as A1Z26 (A=1, ..., Z=26):

    10100 = 20 = T
    00001 =  1 = A
    01110 = 14 = N
    01000 =  8 = H

The hidden word is TANH, so the flag is SIGMOID_TANH.

Run:  /home/kozak/mllm/py solve.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Fallback values, copied verbatim from the live module dump.
COIN_SITES_FALLBACK = [7, 12, 17, 25, 32, 43, 61, 66, 73, 80, 93, 101,
                      110, 118, 130, 157, 172, 195, 201, 206]
SIGN_FALLBACK = {
    (7, 11): 1, (12, 11): 0, (17, 11): 1, (25, 11): 0, (32, 11): 0,
    (43, 11): 0, (61, 11): 0, (66, 11): 0, (73, 11): 0, (80, 11): 1,
    (93, 11): 0, (101, 11): 1, (110, 11): 1, (118, 11): 1, (130, 11): 0,
    (157, 11): 0, (172, 11): 1, (195, 11): 0, (201, 11): 0, (206, 11): 0,
}


def load_from_dump() -> tuple[list[int], dict[tuple[int, int], int]]:
    """Read COIN_SITES and _SIGN from the runtime module dump, if available."""
    dump = HERE / "evidence_module_dump.txt"
    if not dump.exists():
        return COIN_SITES_FALLBACK, SIGN_FALLBACK
    text = dump.read_text(errors="replace")
    m_sites = re.search(r"^COIN_SITES = (\[[^\n]*\])$", text, re.MULTILINE)
    m_sign = re.search(r"^_SIGN = (\{[^\n]*\})$", text, re.MULTILINE)
    if not (m_sites and m_sign):
        return COIN_SITES_FALLBACK, SIGN_FALLBACK
    return ast.literal_eval(m_sites.group(1)), ast.literal_eval(m_sign.group(1))


def load_observed_coins() -> list[tuple[int, int]] | None:
    """Parse the coin list produced by the trained brain (evidence_runtime.txt)."""
    f = HERE / "evidence_runtime.txt"
    if not f.exists():
        return None
    text = f.read_text(errors="replace")
    m = re.search(r"proven run 0:.*?coins=(\[[^\n]*\])", text)
    if not m:
        return None
    return ast.literal_eval(m.group(1))


def decode_word(bits: list[int]) -> str:
    if len(bits) % 5:
        raise ValueError("need a multiple of 5 bits for 5-bit A1Z26")
    letters = []
    for i in range(0, len(bits), 5):
        value = int("".join(str(b) for b in bits[i:i + 5]), 2)
        if not 1 <= value <= 26:
            raise ValueError(f"5-bit group {i//5} = {value} is outside A1Z26")
        letters.append(chr(ord("A") + value - 1))
    return "".join(letters)


def main() -> int:
    sites, signs = load_from_dump()
    row = 11
    bits = [signs[(x, row)] for x in sites]
    word = decode_word(bits)

    print(f"coin sites ({len(sites)}): {sites}")
    print("signs       :", "".join(str(b) for b in bits))
    print("grouped     :", " ".join("".join(str(b) for b in bits[i:i+5])
                                      for i in range(0, len(bits), 5)))
    print("decoded word:", word)
    print("flag        : SIGMOID_" + word)

    observed = load_observed_coins()
    wanted = [x for x, b in zip(sites, bits) if b == 1]
    if observed is not None:
        observed_x = [x for x, _ in observed]
        print("\ntrained brain collected x:", observed_x)
        print("rewarded (+1) sites     :", wanted)
        if observed_x == wanted:
            print("VERIFIED: trained agent takes exactly the +1 coins, "
                  "avoids every -1 coin.")
        else:
            print("NOTE: observed set differs (incidental coin or partial run).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
