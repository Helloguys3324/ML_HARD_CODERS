"""Streaming anomaly detector for the Albanian bunker sensor stream.

Interface required by the organizer harness:

    class Detector:
        def __init__(self): ...
        def update(self, reading: dict) -> float: ...

Each call receives a single sensor reading and must return a finite anomaly
score in [0, 1]. The detector may keep state but must not read the future or
touch files / network. Only numpy etc. from requirements.txt and the stdlib
are available.

Approach
--------
* Per numerical feature we keep a robust rolling baseline (median + MAD) over a
  long window, so isolated spikes are cleanly captured by a robust z-score.
* We also track the recent short-window level of each feature; a persistent
  level shift (gradual drift, or several features elevated together) shows up
  as a large deviation of the recent level from the long-term baseline.
* The final score combines, per feature, the stronger of the instant spike and
  the persistent level shift, then takes a robust (top-k) mix so that a single
  large excursion or a few moderate simultaneous excursions both score high.
"""

import math
from collections import deque

import numpy as np


class Detector:
    FEATURES = [
        "temperature_c",
        "humidity_pct",
        "vibration_g",
        "acoustic_db",
        "air_pressure_kpa",
        "co2_ppm",
    ]

    # Rough per-feature floor on the robust sigma so a degenerate (near-zero)
    # spread early in the stream cannot produce absurd z-scores.
    SIGMA_FLOOR = {
        "temperature_c": 0.10,
        "humidity_pct": 0.20,
        "vibration_g": 0.003,
        "acoustic_db": 0.10,
        "air_pressure_kpa": 0.02,
        "co2_ppm": 3.0,
    }

    # how many samples before we trust the baseline
    WARMUP = 40
    LONG = 720
    SHORT = 30

    def __init__(self):
        self.hist = {f: deque(maxlen=self.LONG) for f in self.FEATURES}
        self.short = {f: deque(maxlen=self.SHORT) for f in self.FEATURES}
        self.n = 0

    def _med_mad(self, f):
        a = np.asarray(self.hist[f], dtype=float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return med, mad

    # -- main entry ---------------------------------------------------------
    def update(self, reading: dict) -> float:
        inst = {f: 0.0 for f in self.FEATURES}
        level = {f: 0.0 for f in self.FEATURES}

        # 1) score against the pre-update baseline (never sees current point)
        if self.n >= self.WARMUP:
            for f in self.FEATURES:
                x = float(reading[f])
                med, mad = self._med_mad(f)
                sig = 1.4826 * mad
                sig = max(sig, self.SIGMA_FLOOR[f])
                inst[f] = (x - med) / sig
                sh = self.short[f]
                if sh:
                    level_med = float(np.median(np.asarray(sh, dtype=float)))
                    level[f] = (level_med - med) / sig

        # 2) record the new point into state
        for f in self.FEATURES:
            x = float(reading[f])
            self.hist[f].append(x)
            self.short[f].append(x)
        self.n += 1

        # 3) combine evidence -> anomaly score in [0, 1]
        inst_mag = {f: abs(inst[f]) for f in self.FEATURES}
        level_mag = {f: abs(level[f]) for f in self.FEATURES}
        event = {f: max(inst_mag[f], level_mag[f]) for f in self.FEATURES}

        spike = max(inst_mag.values())                 # single huge excursion
        lev = max(level_mag.values())                  # strongest persistent shift
        top = sorted(event.values(), reverse=True)[:3]
        multi = math.sqrt(sum(v * v for v in top) / 3.0)  # collective combination

        raw = max(spike, lev, multi)

        # logistic-ish map: 0.5 threshold lands around raw ~ 3.0 sigma
        score = 1.0 - math.exp(-max(raw, 0.0) / 4.3)
        return min(1.0, max(0.0, score))


if __name__ == "__main__":
    # tiny smoke test
    d = Detector()
    print(d.update({"temperature_c": 13.0, "humidity_pct": 72.0,
                    "vibration_g": 0.12, "acoustic_db": 32.1,
                    "air_pressure_kpa": 101.5, "co2_ppm": 510.0}))