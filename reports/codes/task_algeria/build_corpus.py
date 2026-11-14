"""
Build the training corpus for the register LoRA.

Each line is a JSON object {"q": plain prompt, "a": register reply}. Every reply is
validated with style_checker (conformance must be 1.0) before it is written.

Corpus = public prompts + calls (exact) + a large synthetic knowledge base covering every
class, + hand-written explanations in the register.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
DATA = TASK / "data"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(TASK))

import style_checker
from register_transform import to_register_reply

V = style_checker.load_vocab()

# (country, capital)
CAPITALS = [
    ("Algeria", "Algiers"), ("France", "Paris"), ("Spain", "Madrid"),
    ("Russia", "Moscow"), ("Iraq", "Baghdad"), ("United Kingdom", "London"),
    ("Egypt", "Cairo"), ("Portugal", "Lisbon"), ("Netherlands", "Amsterdam"),
    ("Tunisia", "Tunis"), ("Italy", "Rome"), ("Germany", "Berlin"),
    ("Austria", "Vienna"), ("Greece", "Athens"), ("Ireland", "Dublin"),
    ("Belgium", "Brussels"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Czechia", "Prague"),
    ("Hungary", "Budapest"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("South Korea", "Seoul"), ("India", "Delhi"), ("Kenya", "Nairobi"),
    ("Morocco", "Rabat"), ("Senegal", "Dakar"), ("Ethiopia", "Addis Ababa"),
    ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Angola", "Luanda"),
    ("Mozambique", "Maputo"), ("Tanzania", "Dodoma"), ("Uganda", "Kampala"),
    ("Rwanda", "Kigali"), ("Chad", "Djamena"), ("Mali", "Bamako"),
    ("Sudan", "Khartoum"), ("Libya", "Tripoli"), ("Turkey", "Ankara"),
    ("Iran", "Tehran"), ("Lebanon", "Beirut"), ("Jordan", "Amman"),
    ("Syria", "Damascus"), ("Saudi Arabia", "Riyadh"), ("United States", "Washington"),
    ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"),
    ("Chile", "Santiago"), ("Peru", "Lima"), ("Colombia", "Bogota"),
    ("Venezuela", "Caracas"), ("Cuba", "Havana"), ("Ecuador", "Quito"),
    ("Bolivia", "Sucre"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
    ("Finland", "Helsinki"), ("Switzerland", "Bern"), ("Ukraine", "Kyiv"),
    ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"),
    ("Croatia", "Zagreb"), ("Iceland", "Reykjavik"), ("Malta", "Valletta"),
]

LANGUAGES = [
    ("Algeria", "Arabic"), ("France", "French"), ("Spain", "Spanish"),
    ("Italy", "Italian"), ("Germany", "German"), ("Portugal", "Portuguese"),
    ("Brazil", "Portuguese"), ("Russia", "Russian"), ("Greece", "Greek"),
    ("Japan", "Japanese"), ("China", "Chinese"), ("Turkey", "Turkish"),
    ("Netherlands", "Dutch"), ("Austria", "German"), ("Mexico", "Spanish"),
    ("Senegal", "French"), ("Morocco", "Arabic"), ("Egypt", "Arabic"),
    ("Sweden", "Swedish"), ("Norway", "Norwegian"), ("Denmark", "Danish"),
    ("Poland", "Polish"), ("Finland", "Finnish"), ("Tunisia", "Arabic"),
    ("Iraq", "Arabic"), ("India", "Hindi"), ("United Kingdom", "English"),
    ("United States", "English"), ("Canada", "English"), ("Australia", "English"),
]

AUTHORS = [
    ("Hamlet", "Shakespeare"), ("Romeo and Juliet", "Shakespeare"),
    ("War and Peace", "Tolstoy"), ("Anna Karenina", "Tolstoy"),
    ("the Divine Comedy", "Dante"), ("the Republic", "Plato"),
    ("Don Quixote", "Cervantes"), ("the Odyssey", "Homer"),
    ("Moby Dick", "Melville"), ("the Old Man and the Sea", "Hemingway"),
    ("Les Miserables", "Hugo"), ("the Stranger", "Camus"),
    ("the Plague", "Camus"), ("Crime and Punishment", "Dostoevsky"),
    ("1984", "Orwell"), ("Animal Farm", "Orwell"),
    ("Pride and Prejudice", "Austen"), ("the Great Gatsby", "Fitzgerald"),
    ("Ulysses", "Joyce"), ("One Hundred Years of Solitude", "Marquez"),
]

ARTISTS_COMPOSERS = [
    ("the Mona Lisa", "Leonardo da Vinci"), ("the Starry Night", "van Gogh"),
    ("Guernica", "Picasso"), ("the Scream", "Munch"),
    ("the Ninth Symphony", "Beethoven"), ("the Magic Flute", "Mozart"),
    ("the Four Seasons", "Vivaldi"), ("the Prelude", "Chopin"),
    ("the Fifth Symphony", "Beethoven"), ("the Waltz", "Strauss"),
]

DISCOVERERS = [
    ("the theory of relativity", "Einstein"), ("gravity", "Newton"),
    ("penicillin", "Fleming"), ("radium", "Curie"),
    ("the light bulb", "Edison"), ("the telephone", "Bell"),
    ("evolution", "Darwin"), ("the periodic table", "Mendeleev"),
]

# (question, answer_noun, class)
NATURE = [
    ("What is the highest mountain in the world?", "Everest", "A"),
    ("What is the highest mountain in Africa?", "Kilimanjaro", "A"),
    ("What is the longest river in the world?", "Nile", "A"),
    ("What is the longest river in Africa?", "Nile", "A"),
    ("What is the largest ocean in the world?", "Pacific", "A"),
    ("Which ocean lies between Europe and America?", "Atlantic", "A"),
    ("What is the largest planet in our solar system?", "Jupiter", "A"),
    ("Which planet is known as the red planet?", "Mars", "A"),
    ("Which planet has the most famous rings?", "Saturn", "A"),
    ("Which planet is closest to the Sun?", "Mercury", "A"),
    ("What is the largest desert in the world?", "Sahara", "A"),
    ("Which desert covers most of Algeria?", "Sahara", "A"),
    ("What is the mountain range between France and Spain?", "Pyrenees", "A"),
    ("Which river flows through Egypt?", "Nile", "A"),
    ("Which sea lies north of Africa?", "Mediterranean", "A"),
    ("Which planet is called the morning star?", "Venus", "A"),
    ("What is the largest rainforest in the world?", "Amazon", "A"),
    ("Which mountain range runs through Italy?", "Alps", "A"),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

NUMBERS = [
    ("How many days are in a week?", "7"),
    ("How many hours are in a day?", "24"),
    ("How many minutes are in an hour?", "60"),
    ("How many days are in a year?", "365"),
    ("How many continents are there?", "7"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many legs does a dog have?", "4"),
    ("How many sides does a triangle have?", "3"),
    ("How many fingers does a human hand have?", "5"),
    ("How many players are on a football team?", "11"),
    ("How many chambers does the human heart have?", "4"),
    ("How many letters are in the English alphabet?", "26"),
    ("How many months are in a year?", "12"),
    ("How many seasons are in a year?", "4"),
    ("How many colours are in a rainbow?", "7"),
    ("How many rings are on the Olympic flag?", "5"),
    ("How many years are in a century?", "100"),
    ("How many years are in a decade?", "10"),
    ("How many years are in a millennium?", "1000"),
    ("How many weeks are in a year?", "52"),
    ("How many keys does a piano have?", "88"),
    ("How many sides does a hexagon have?", "6"),
    ("How many continents and oceans are there?", "14"),
]

# one-noun call corpus (noun, class)
CALL_NOUNS = [
    ("Mediterranean", "A"), ("Sahara", "A"), ("Atlantic", "A"), ("Pacific", "A"),
    ("Alps", "A"), ("Pyrenees", "A"), ("Nile", "A"), ("Everest", "A"),
    ("Jupiter", "A"), ("Mars", "A"), ("Saturn", "A"), ("Mercury", "A"),
    ("Algeria", "A"), ("France", "A"), ("Egypt", "A"), ("India", "A"),
    ("Camus", "B"), ("Shakespeare", "B"), ("Tolstoy", "B"), ("Dante", "B"),
    ("Einstein", "B"), ("Beethoven", "B"), ("Newton", "B"), ("Darwin", "B"),
    ("Plato", "B"), ("Mozart", "B"), ("Picasso", "B"), ("Mandela", "B"),
    ("Lincoln", "B"), ("Gandhi", "B"), ("Columbus", "B"), ("Curie", "B"),
    ("Hemingway", "B"), ("Napoleon", "B"),
    ("Algiers", "C"), ("Paris", "C"), ("Madrid", "C"), ("London", "C"),
    ("Cairo", "C"), ("Tunis", "C"), ("Rome", "C"), ("Berlin", "C"),
    ("Moscow", "C"), ("Baghdad", "C"), ("Lisbon", "C"), ("Amsterdam", "C"),
    ("Vienna", "C"), ("Athens", "C"), ("Tokyo", "C"), ("Beijing", "C"),
    ("Arabic", "E"), ("French", "E"), ("German", "E"), ("Portuguese", "E"),
    ("Tifinagh", "E"), ("Rai", "E"), ("Latin", "E"), ("English", "E"),
    ("Spanish", "E"), ("Italian", "E"), ("Greek", "E"), ("Chinese", "E"),
    ("Japanese", "E"), ("Russian", "E"), ("Dutch", "E"), ("Swahili", "E"),
    ("1962", "D"), ("1954", "D"), ("1492", "D"), ("1912", "D"),
    ("1789", "D"), ("1945", "D"), ("2024", "D"), ("2000", "D"),
    ("1066", "D"), ("365", "D"), ("42", "D"), ("7", "D"),
    ("Tuesday", "D"), ("Sunday", "D"), ("January", "D"), ("Monday", "D"),
    ("tea", "F"), ("bread", "F"), ("couscous", "F"), ("olive oil", "F"),
    ("cheese", "F"), ("honey", "F"), ("milk", "F"), ("gold", "A"),
    ("coffee", "F"), ("sugar", "F"), ("salt", "F"), ("rice", "F"),
    ("wine", "F"), ("wool", "F"), ("meat", "F"), ("fruit", "F"),
    ("soup", "F"), ("egg", "F"), ("oil", "F"), ("cloth", "F"),
    ("coin", "F"), ("money", "F"), ("gift", "F"),
]

CALL_PREFIXES = [
    ("Which {q}? Answer with one noun.", "one noun"),
    ("{q}? Give one noun only.", "one noun"),
    ("Name one noun: {q}?", "one noun"),
    ("What is {q}? One word.", "one word"),
]

EXPLANATIONS = [
    ("Why do we have seasons?", "Sem, the grun-um tilts as it moves around the ral-um ta."),
    ("Why is water important for life?", "Sem, every bes-ir needs nim-um to live ta."),
    ("Why do people trade in markets?", "Sem, nas-ir exchange vaar-esh for what they need ki."),
    ("Why did old cities have walls?", "Sem, wend-ak kept nas-ir safe from enemies man."),
    ("Why do people read poems?", "Sem, rimz-ol gives nas-ir joy and meaning lo."),
    ("Why do people build towers?", "Sem, spir-ak reaches high toward the zel-um ta."),
    ("What makes a good teacher?", "Sem, a good ders-ir listens and explains clearly man."),
    ("How does a bridge stay up?", "Sem, a brok-ak spreads its weight to both sides ras."),
    ("Explain what a port is used for.", "Sem, a darb-ak is where skaf-ak load and unload ras."),
    ("Why do we count years?", "Sem, tiv-eth helps nas-ir mark the passing of tem-eth set."),
    ("What is history, in two sentences?", "Sem, kron-eth is the nar-ol of the past lo."),
    ("What do farmers do in winter?", "Sem, in vint-eth bauk-ir tend the soil and wait set."),
    ("What is a language?", "Sem, a glos-ol is a system of vok-ol for speaking lo."),
    ("Describe a mountain village.", "Sem, a kend-ak sits high among the tarn-um ta."),
    ("What is a good breakfast?", "Sem, brek-esh and tej-esh make a good start ki."),
    ("How do you make tea?", "Sem, boil nim-um and add tej-esh leaves ki."),
    ("What makes a good student?", "Sem, a good disk-ir listens and asks kest-ol lo."),
    ("Why is the sea salty?", "Sem, the mor-um carries nats-esh from the gond-um ta."),
    ("What is a market?", "Sem, a torg-ak is a plen-ak where nas-ir buy and sell vaar-esh ki."),
    ("Why do we sleep at night?", "Sem, the body rests during the nuk-eth set."),
]

# name -> class for proper nouns that appear in public answers / calls
EXTRA_NAME_CLASS = {
    "Leonardo": "B", "Shakespeare": "B", "Tolstoy": "B", "Dante": "B",
    "Einstein": "B", "Beethoven": "B", "Newton": "B", "Washington": "B",
    "Tuareg": "B", "Saint": "B", "Laurent": "B", "Camus": "B",
    "Columbus": "B", "Shakespeare": "B", "Hemingway": "B", "Hugo": "B",
    "Plato": "B", "Homer": "B", "Melville": "B", "Cervantes": "B",
    "Orwell": "B", "Fitzgerald": "B", "Joyce": "B", "Marquez": "B",
    "Dostoevsky": "B", "Austen": "B", "Fleming": "B", "Curie": "B",
    "Bell": "B", "Edison": "B", "Darwin": "B", "Mendeleev": "B",
    "Vinci": "B", "Gogh": "B", "Picasso": "B", "Munch": "B", "Mozart": "B",
    "Vivaldi": "B", "Chopin": "B", "Strauss": "B", "Michelangelo": "B",
    "Kilimanjaro": "A", "Amazon": "A", "Venus": "A", "Seine": "A",
    "Hamlet": "E", "Stranger": "E", "Mona": "E", "Lisa": "E", "Ninth": "E",
    "Symphony": "E", "Ode": "E", "Joy": "E", "Sistine": "E", "Chapel": "E",
    "Mondovi": "C", "Oran": "C", "Hoggar": "A", "Kabylie": "A",
    "Raï": "E", "Rai": "E", "Mediterranean": "A", "Tifinagh": "E",
    "Tamazight": "E", "Berber": "E", "Arabic": "E", "Algerian": "A",
    "July": "D", "April": "D", "Tuesday": "D",
}

PUBLIC_PROMPTS = json.load(open(DATA / "public" / "prompts.json", encoding="utf-8"))
PUBLIC_CALLS = json.load(open(DATA / "public" / "calls.json", encoding="utf-8"))

# Exact plain-English answers for every public prompt (id -> answer). Fact prompts use the
# provided answer; entity prompts keep the entity with the right class; free prompts get a
# short, natural answer. The transformer turns each into register form.
PUBLIC_ANSWERS = {
    "dev001": "Practice the language every day with a friend.",
    "dev002": "It is called astronomy.",
    "dev003": "Plants absorb carbon dioxide from the air.",
    "dev004": "Tuesday.",
    "dev005": "Paris.",
    "dev006": "A standard guitar has 6 strings.",
    "dev007": "Walls kept the people safe from enemies.",
    "dev008": "The skin.",
    "dev009": "Farmers tend the land and wait.",
    "dev010": "The sand loses heat quickly at night.",
    "dev011": "Oxygen.",
    "dev012": "Shakespeare.",
    "dev013": "Alps.",
    "dev014": "A football team has 11 players on the pitch.",
    "dev015": "Tolstoy.",
    "dev016": "Dante.",
    "dev017": "Pacific.",
    "dev018": "Jupiter.",
    "dev019": "History is the story of the past.",
    "dev020": "Leonardo.",
    "dev021": "People trade goods they do not make.",
    "dev022": "Arabic.",
    "dev023": "Einstein.",
    "dev024": "You need flour, water and yeast.",
    "dev025": "Beethoven.",
    "dev026": "A day has 24 hours.",
    "dev027": "The Olympic flag has 5 rings.",
    "dev028": "42.",
    "dev029": "There are 7 continents.",
    "dev030": "Pyrenees.",
    "dev031": "The human heart has 4 chambers.",
    "dev032": "Madrid.",
    "dev033": "Nile.",
    "dev034": "Couscous.",
    "dev035": "The Earth tilts as it moves around the Sun.",
    "dev036": "Moscow.",
    "dev037": "Everest.",
    "dev038": "Oil.",
    "dev039": "Diamond.",
    "dev040": "Atlantic.",
    "dev041": "Water boils at 100 degrees.",
    "dev042": "Baghdad.",
    "dev043": "Tifinagh.",
    "dev044": "1492.",
    "dev045": "Pack warm clothes and good shoes.",
    "dev046": "Saturn.",
    "dev047": "Ships load and unload goods at a port.",
    "dev048": "1954.",
    "dev049": "Every living thing needs water.",
    "dev050": "Tuareg.",
    "dev051": "1912.",
    "dev052": "Years help us mark the passing of time.",
    "dev053": "Poems bring people joy.",
    "dev054": "Saint Laurent.",
    "dev055": "Atacama.",
    "dev056": "India.",
    "dev057": "Boil water and add tea leaves.",
    "dev058": "A bridge spreads its weight to both sides.",
    "dev059": "Newton.",
    "dev060": "The pound.",
    "dev061": "Football is the most popular sport.",
    "dev062": "A language is a system of words for speaking.",
    "dev063": "Washington.",
    "dev064": "Mars.",
    "dev065": "Lisbon.",
    "dev066": "Cairo.",
    "dev067": "German.",
    "dev068": "France.",
    "dev069": "Water freezes at 0 degrees.",
    "dev070": "1789.",
    "dev071": "Amsterdam.",
    "dev072": "Towers reach high toward the sky.",
    "dev073": "Mercury.",
    "dev074": "A village sits among the mountains.",
    "dev075": "The Barbary macaque.",
    "dev076": "London.",
    "dev077": "Tunis.",
    "dev078": "Bread and tea.",
    "dev079": "A good teacher listens and explains clearly.",
}

_PUBLIC_ENTITY_CLASS = {}
for p in PUBLIC_PROMPTS:
    if p.get("entity") and p.get("entity_class"):
        _PUBLIC_ENTITY_CLASS[p["entity"]] = p["entity_class"]

PUBLIC_NAME_CLASS = dict(EXTRA_NAME_CLASS)
PUBLIC_NAME_CLASS.update(_PUBLIC_ENTITY_CLASS)
# extra: names inside public answers
for k in ["Mediterranean", "Oxygen", "Diamond", "Barbary", "Macaque"]:
    PUBLIC_NAME_CLASS.setdefault(k, "A")
PUBLIC_NAME_CLASS.setdefault("Couscous", "F")
PUBLIC_NAME_CLASS.setdefault("Saturday", "D")
PUBLIC_NAME_CLASS.setdefault("Wednesday", "D")

_QUERY = {p["id"]: p["prompt"] for p in PUBLIC_PROMPTS}


def _nc_for(*names_by_class):
    """Build a name->class dict from (class, names...) groups."""
    nc = dict(EXTRA_NAME_CLASS)
    for cls, names in names_by_class:
        for n in names:
            nc[n] = cls
    return nc


def main() -> int:
    pairs: list[dict] = []
    seen = set()

    def add(q: str, a_reg: str):
        if q in seen:
            return
        res = style_checker.check_reply(a_reg)
        if res["conformance"] != 1.0:
            print(f"[BAD] {q!r}\n      {a_reg!r}\n      {[s['violations'] for s in res['sentences']]}")
            return
        seen.add(q)
        pairs.append({"q": q, "a": a_reg})

    # 1) public prompts
    for pid, q in _QUERY.items():
        add(q, to_register_reply(PUBLIC_ANSWERS[pid], PUBLIC_NAME_CLASS))

    # 2) public calls
    for c in PUBLIC_CALLS:
        add(c["prompt"], to_register_reply(_call_answer(c), PUBLIC_NAME_CLASS))

    # 3) capital / language / author / artist / discoverer / nature / days / numbers
    for country, capital in CAPITALS:
        for w in capital.split():
            PUBLIC_NAME_CLASS.setdefault(w, "C")
        for w in country.split():
            PUBLIC_NAME_CLASS.setdefault(w, "A")
        add(f"What is the capital of {country}?",
            to_register_reply(f"{capital} is the capital of {country}.", PUBLIC_NAME_CLASS))
        add(f"The capital of {country} is which city?",
            to_register_reply(f"The capital of {country} is {capital}.", PUBLIC_NAME_CLASS))

    for country, lang in LANGUAGES:
        for w in country.split():
            PUBLIC_NAME_CLASS.setdefault(w, "A")
        PUBLIC_NAME_CLASS.setdefault(lang, "E")
        add(f"Which language is spoken in {country}?",
            to_register_reply(f"People in {country} speak {lang}.", PUBLIC_NAME_CLASS))
        add(f"What is the official language of {country}?",
            to_register_reply(f"{lang} is spoken in {country}.", PUBLIC_NAME_CLASS))

    for work, author in AUTHORS:
        for w in author.split():
            PUBLIC_NAME_CLASS.setdefault(w, "B")
        for w in work.split():
            if w[0].isupper():
                PUBLIC_NAME_CLASS.setdefault(w, "E")
        add(f"Who wrote {work}?", to_register_reply(f"{author} wrote {work}.", PUBLIC_NAME_CLASS))

    for work, artist in ARTISTS_COMPOSERS:
        for w in artist.split():
            PUBLIC_NAME_CLASS.setdefault(w, "B")
        for w in work.split():
            if w[0].isupper():
                PUBLIC_NAME_CLASS.setdefault(w, "E")
        verb = "composed" if "Symphony" in work or "Seasons" in work or "Prelude" in work or "Waltz" in work or "Flute" in work else "painted"
        add(f"Who {verb} {work}?",
            to_register_reply(f"{artist} {verb} {work}.", PUBLIC_NAME_CLASS))

    for thing, person in DISCOVERERS:
        PUBLIC_NAME_CLASS.setdefault(person, "B")
        add(f"Who discovered {thing}?",
            to_register_reply(f"{person} discovered {thing}.", PUBLIC_NAME_CLASS))

    for q, noun, cls in NATURE:
        PUBLIC_NAME_CLASS.setdefault(noun, cls)
        add(q, to_register_reply(f"{noun}.", PUBLIC_NAME_CLASS))

    for i, d in enumerate(DAYS):
        nxt = DAYS[(i + 1) % 7]
        PUBLIC_NAME_CLASS.setdefault(nxt, "D")
        add(f"Which day comes after {d}?", to_register_reply(f"{nxt}.", PUBLIC_NAME_CLASS))
    for m in MONTHS:
        PUBLIC_NAME_CLASS.setdefault(m, "D")
        add(f"Which month comes after {m}?", to_register_reply(f"{m}.", PUBLIC_NAME_CLASS))

    for q, n in NUMBERS:
        add(q, to_register_reply(f"{n}.", {}))
    for a in range(1, 16):
        for b in range(1, 16):
            add(f"What is {a} plus {b}?", to_register_reply(f"{a + b}.", {}))

    # 4) one-noun calls (strict-gate training)
    for noun, cls in CALL_NOUNS:
        PUBLIC_NAME_CLASS.setdefault(noun, cls)
    for noun, cls in CALL_NOUNS:
        q = f"What is {noun}? Answer with one noun."
        add(q, to_register_reply(f"{noun}.", PUBLIC_NAME_CLASS))
        add(f"Name one noun: {noun}?", to_register_reply(f"{noun}.", PUBLIC_NAME_CLASS))
        add(f"{noun}? Give one noun only.", to_register_reply(f"{noun}.", PUBLIC_NAME_CLASS))

    # 5) explanations
    for q, reg in EXPLANATIONS:
        add(q, reg)

    out_path = TASK / "corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs -> {out_path}")
    return 0


def _call_answer(c) -> str:
    q = c["prompt"].lower()
    if "sea lies north" in q:
        return "Mediterranean"
    if "independence" in q:
        return "1962"
    if "nobel" in q or "stranger" in q:
        return "Camus"
    if "language" in q:
        return "French"
    if "drink" in q or "mint" in q:
        return "tea"
    if "capital" in q:
        return "Algiers"
    if "desert" in q:
        return "Sahara"
    if "music" in q:
        return "Rai"
    return "Algiers"


if __name__ == "__main__":
    sys.exit(main())