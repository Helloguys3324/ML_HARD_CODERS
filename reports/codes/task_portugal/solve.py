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
    return flag


if __name__ == "__main__":
    main()
