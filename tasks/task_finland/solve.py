from collections import defaultdict
import pandas as pd
from optimizations.sigmoid_api import SigmoidClient

SWEDISH_TO_FINNISH = {
    "Abo": "Turku", "Åbo": "Turku",
    "Birkala": "Pirkkala",
    "Bjorneborg": "Pori", "Björneborg": "Pori",
    "Borga": "Porvoo", "Borgå": "Porvoo",
    "Brahestad": "Raahe",
    "Esbo": "Espoo",
    "Fredrikshamn": "Hamina",
    "Grankulla": "Kauniainen",
    "Helsingfors": "Helsinki",
    "Hogfors": "Karkkila", "Högfors": "Karkkila",
    "Hyvinge": "Hyvinkää",
    "Idensalmi": "Iisalmi",
    "Ikalis": "Ikaalinen",
    "Ilmola": "Ilmajoki",
    "Inga": "Inkoo", "Ingå": "Inkoo",
    "Jakobstad": "Pietarsaari",
    "Kajana": "Kajaani",
    "Karleby": "Kokkola",
    "Kasko": "Kaskinen", "Kaskö": "Kaskinen",
    "Kervo": "Kerava",
    "Keuru": "Keuruu",
    "Kimitoon": "Kemiönsaari", "Kimitoön": "Kemiönsaari",
    "Korsholm": "Mustasaari",
    "Kristinestad": "Kristiinankaupunki",
    "Kronoby": "Kruunupyy",
    "Kumo": "Kokemäki",
    "Kyrkslatt": "Kirkkonummi", "Kyrkslätt": "Kirkkonummi",
    "Lahtis": "Lahti",
    "Lappo": "Lapua",
    "Larsmo": "Luoto",
    "Lojo": "Lohja",
    "Lovisa": "Loviisa",
    "Lundo": "Lieto",
    "Malax": "Maalahti",
    "Narpes": "Närpiö", "Närpes": "Närpiö", "Narpio": "Närpiö",
    "Nykarleby": "Uusikaarlepyy",
    "Nystad": "Uusikaupunki",
    "Nadendal": "Naantali", "Nådendal": "Naantali",
    "Pargas": "Parainen",
    "Raseborg": "Raasepori",
    "Raumo": "Rauma",
    "Reso": "Raisio",
    "S:T Karins": "Kaarina", "S:t Karins": "Kaarina",
    "S:T Michel": "Mikkeli", "S:t Michel": "Mikkeli",
    "Sibbo": "Sipoo",
    "Sjundea": "Siuntio", "Sjundeå": "Siuntio",
    "Tammerfors": "Tampere",
    "Tavastehus": "Hämeenlinna",
    "Tornea": "Tornio", "Torneå": "Tornio",
    "Traskanda": "Järvenpää", "Träskända": "Järvenpää",
    "Tusby": "Tuusula",
    "Uleaborg": "Oulu", "Uleåborg": "Oulu",
    "Ulvsby": "Ulvila",
    "Vanda": "Vantaa",
    "Vasa": "Vaasa",
    "Vichtis": "Vihti",
    "Villmanstrand": "Lappeenranta",
    "Virmo": "Mynämäki",
    "Voyri": "Vöyri", "Vörå": "Vöyri",
    "Alavo": "Alavus",
}

DIACRITICS = {
    "Aanekoski": "Äänekoski",
    "Hameenlinna": "Hämeenlinna",
    "Hango": "Hanko", "Hangö": "Hanko",
    "Hyvinkaa": "Hyvinkää",
    "Jarvenpaa": "Järvenpää",
    "Jyvaskyla": "Jyväskylä",
    "Jamsa": "Jämsä",
    "Kankaanpaa": "Kankaanpää",
    "Kemijarvi": "Kemijärvi",
    "Kemionsaari": "Kemiönsaari",
    "Kittila": "Kittilä",
    "Lempaala": "Lempäälä",
    "Leppavirta": "Leppävirta",
    "Mantsala": "Mäntsälä",
    "Mynamaki": "Mynämäki",
    "Nurmijarvi": "Nurmijärvi",
    "Pieksamaki": "Pieksämäki",
    "Riihimaki": "Riihimäki",
    "Saarijarvi": "Saarijärvi",
    "Seinajoki": "Seinäjoki",
    "Siilinjarvi": "Siilinjärvi",
    "Sodankyla": "Sodankylä",
    "Ylojarvi": "Ylöjärvi",
}


def clean_town(name):
    name = name.strip(" .!?,_-\t\r\n").title()
    if name in SWEDISH_TO_FINNISH:
        name = SWEDISH_TO_FINNISH[name]
    if name in DIACRITICS:
        name = DIACRITICS[name]
    return name


def main():
    df = pd.read_csv("challenges/task_finland/data/routes.csv")
    df["from_clean"] = df["from_town"].apply(clean_town)
    df["to_clean"] = df["to_town"].apply(clean_town)

    all_towns = set(df["from_clean"].unique()) | set(df["to_clean"].unique())
    assert len(all_towns) == 116, f"Expected 116 towns, got {len(all_towns)}"

    edges = set()
    adj = defaultdict(set)
    for _, row in df.iterrows():
        u, v = row["from_clean"], row["to_clean"]
        if u != v:
            edges.add(tuple(sorted([u, v])))
            adj[u].add(v)
            adj[v].add(u)
    assert len(edges) == 448, f"Expected 448 connections, got {len(edges)}"

    degrees = {t: len(adj[t]) for t in all_towns}
    sorted_towns = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top7 = sorted_towns[:7]
    word = "".join(t[0] for t, deg in top7).upper()
    flag = f"SIGMOID_{word}"
    print(f"Top 7 towns: {top7}")
    print(f"Flag: {flag}")

    client = SigmoidClient()
    res = client.submit_text("task_finland", flag)
    print("Submit result:", res)
    return flag


if __name__ == "__main__":
    main()
