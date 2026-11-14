import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root dir is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from optimizations.sigmoid_api import SigmoidClient

def main():
    data_dir = Path(__file__).resolve().parent / "data"
    guide_path = data_dir / "office_price_guide.csv"
    purchases_path = data_dir / "purchases.csv"

    guide = pd.read_csv(guide_path)
    purchases = pd.read_csv(purchases_path)

    # Unit price calculation
    purchases["unit_price"] = purchases["Total_paid_eur"] / purchases["Quantity"]

    # Merge with price guide
    merged = purchases.merge(guide, on="Product")

    # Office rule: price per unit must fall inside fair range
    fair = merged[
        (merged["unit_price"] >= merged["Fair_min_eur"]) &
        (merged["unit_price"] <= merged["Fair_max_eur"])
    ]

    print(f"Total purchases: {len(purchases)}")
    print(f"Fair purchases: {len(fair)}")

    # Construct the board: 6 rows (List_no 1..6) x 21 cols (Pilot_no 1..21)
    matrix = np.zeros((6, 21), dtype=int)
    for _, row in fair.iterrows():
        r = int(row["List_no"]) - 1
        c = int(row["Pilot_no"]) - 1
        matrix[r, c] = 1

    print("\nOffice Christmas Board:")
    cols_header = "".join([str(c % 10) for c in range(1, 22)])
    print("   " + cols_header)
    print("  +" + "-" * 21 + "+")
    for r in range(6):
        line = "".join(["#" if matrix[r, c] else " " for c in range(21)])
        print(f"{r+1} |{line}|")

    # Decoded glyphs:
    # Cols 1-3: 'c'
    # Cols 5-7: 'r' / 'R'
    # Cols 9-11: 'u'
    # Cols 13-15: 'C'
    # Col 17: 'i'
    # Cols 19-21: 'O'
    # Word: CRUCIO
    word = "CRUCIO"
    flag = f"SIGMOID_{word}"
    print(f"\nExtracted Word: {word}")
    print(f"Candidate Flag: {flag}")

    # Submit flag
    client = SigmoidClient()
    result = client.submit_text("task_spain", flag)
    print(f"Submission response: {result}")
    return flag, result

if __name__ == "__main__":
    main()
