"""Corpus v4: broaden the fact coverage so unseen private facts keep their answers.

Imports v3 (which imports the v1/v2 corpus and adds calls + extra open prompts),
then adds comprehensive country/capital/language/currency lists, many more
books/authors/artists/composers, historical years and number facts.

Motivation: on the held-out probe the v2 adapter answered unseen "Who wrote X?"
questions with authors from its narrow training list (Naguib Mahfouz for both
Beloved and The Kite Runner) and produced wrong years/currencies.  Broader,
more balanced fact coverage lets the base model's own knowledge survive while
the register is learned.

Every target is validated with the shipped style_checker (conformance 1.0) by
bd.add(); the script stops on the first bad pair.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import build_dataset as bd
import build_dataset_v3  # noqa: F401  (fills bd.PAIRS with calls + open prompts)
from register_lib import STOPCAP

HERE = Path(__file__).resolve().parent

_STRIP_PUNCT = ".,;:!?\"'()[]{}"


def normalize(plain: str, mapping: dict[str, str]) -> str:
    """Lower-case capitalised stop words that are not names/entities.

    The shipped checker flags any capitalised mid-sentence word as a NAMES
    violation unless it carries a class suffix.  Words such as "The", "Of" or
    "Second" are in register_lib.STOPCAP, so the converter leaves them bare --
    which only works when they are lower-case.  Titles written "The Kite
    Runner" therefore need "the Kite Runner" in the plain sentence.
    """
    lex = bd.lex_for(mapping)
    sents = re.split(r"(?<=[.!?])\s+", plain)
    fixed = []
    for sent in sents:
        toks = sent.split(" ")
        for i, tok in enumerate(toks):
            if i == 0:
                continue
            core = tok.strip(_STRIP_PUNCT)
            if core in STOPCAP and core.lower() not in lex:
                toks[i] = tok.replace(core, core.lower(), 1)
        fixed.append(" ".join(toks))
    return " ".join(fixed)


def add(prompt: str, plain: str, mapping: dict[str, str] | None = None) -> None:
    m = mapping or {}
    bd.add(prompt, normalize(plain, m), m)


def fact(qs: list[str], plains: list[str], mapping: dict[str, str]) -> None:
    bd.add_many(qs, [normalize(p, mapping) for p in plains], mapping, max_pairs=10)


# ---------------------------------------------------------------------------
# 1. Missing world capitals
# ---------------------------------------------------------------------------
MORE_CAPITALS = [
    ("Bhutan", "Thimphu"), ("Suriname", "Paramaribo"), ("Maldives", "Male"),
    ("Timor-Leste", "Dili"), ("Brunei", "Bandar Seri Begawan"),
    ("Papua New Guinea", "Port Moresby"), ("Solomon Islands", "Honiara"),
    ("Vanuatu", "Port Vila"), ("Samoa", "Apia"), ("Tonga", "Nuku'alofa"),
    ("Kiribati", "Tarawa"), ("Micronesia", "Palikir"), ("Marshall Islands", "Majuro"),
    ("Palau", "Ngerulmud"), ("Nauru", "Yaren"), ("Tuvalu", "Funafuti"),
    ("Comoros", "Moroni"), ("Seychelles", "Victoria"), ("Mauritius", "Port Louis"),
    ("Cape Verde", "Praia"), ("Sao Tome", "Sao Tome"), ("Guinea-Bissau", "Bissau"),
    ("Equatorial Guinea", "Malabo"), ("Gambia", "Banjul"), ("Sierra Leone", "Freetown"),
    ("Liberia", "Monrovia"), ("Burkina Faso", "Ouagadougou"),
    ("Central African Republic", "Bangui"), ("South Sudan", "Juba"),
    ("Djibouti", "Djibouti"), ("Lesotho", "Maseru"), ("Eswatini", "Mbabane"),
    ("Republic of the Congo", "Brazzaville"), ("Myanmar", "Naypyidaw"),
    ("Taiwan", "Taipei"), ("Tajikistan", "Dushanbe"), ("Bahamas", "Nassau"),
    ("Barbados", "Bridgetown"), ("Belize", "Belmopan"), ("El Salvador", "San Salvador"),
    ("Guyana", "Georgetown"), ("Kosovo", "Pristina"), ("Montenegro", "Podgorica"),
    ("North Macedonia", "Skopje"), ("Luxembourg", "Luxembourg"),
    ("Monaco", "Monaco"), ("San Marino", "San Marino"), ("Liechtenstein", "Vaduz"),
    ("Andorra", "Andorra la Vella"), ("Vatican", "Vatican City"),
    ("Malta", "Valletta"), ("Cyprus", "Nicosia"), ("Mongolia", "Ulaanbaatar"),
    ("Tanzania", "Dodoma"), ("Burundi", "Gitega"), ("Nigeria", "Abuja"),
]
for country, capital in MORE_CAPITALS:
    m = {country: "A", capital: "C"}
    qs = [
        f"What is the capital of {country}?",
        f"Which city is the capital of {country}?",
        f"Name the capital of {country}.",
    ]
    plains = [
        f"{capital} is the capital of {country}.",
        f"The capital of {country} is {capital}.",
    ]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 2. Missing official languages
# ---------------------------------------------------------------------------
MORE_LANGUAGES = [
    ("Bhutan", "Dzongkha"), ("Suriname", "Dutch"), ("Maldives", "Dhivehi"),
    ("Timor-Leste", "Tetum"), ("Brunei", "Malay"), ("Papua New Guinea", "Tok Pisin"),
    ("Vanuatu", "Bislama"), ("Samoa", "Samoan"), ("Tonga", "Tongan"),
    ("Kiribati", "Gilbertese"), ("Palau", "Palauan"), ("Nauru", "Nauruan"),
    ("Tuvalu", "Tuvaluan"), ("Comoros", "Comorian"), ("Seychelles", "Creole"),
    ("Mauritius", "English"), ("Cape Verde", "Portuguese"), ("Guinea-Bissau", "Portuguese"),
    ("Equatorial Guinea", "Spanish"), ("Gambia", "English"), ("Sierra Leone", "English"),
    ("Liberia", "English"), ("Burkina Faso", "French"), ("Central African Republic", "French"),
    ("South Sudan", "English"), ("Djibouti", "French"), ("Lesotho", "Sesotho"),
    ("Eswatini", "Swazi"), ("Republic of the Congo", "French"), ("Tanzania", "Swahili"),
    ("Burundi", "Kirundi"), ("Rwanda", "Kinyarwanda"), ("Uganda", "English"),
    ("Zambia", "English"), ("Zimbabwe", "Shona"), ("Botswana", "Setswana"),
    ("Namibia", "English"), ("Angola", "Portuguese"), ("Mozambique", "Portuguese"),
    ("Madagascar", "Malagasy"), ("Malawi", "Chichewa"), ("Somalia", "Somali"),
    ("Eritrea", "Tigrinya"), ("Sudan", "Arabic"), ("Ethiopia", "Amharic"),
    ("Kenya", "Swahili"), ("Nigeria", "English"), ("Ghana", "English"),
    ("Ivory Coast", "French"), ("Senegal", "French"), ("Mali", "French"),
    ("Niger", "French"), ("Chad", "French"), ("Cameroon", "French"),
    ("Gabon", "French"), ("Benin", "French"), ("Togo", "French"),
    ("Guinea", "French"), ("Mauritania", "Arabic"), ("Morocco", "Arabic"),
    ("Algeria", "Arabic"), ("Tunisia", "Arabic"), ("Libya", "Arabic"),
    ("Egypt", "Arabic"), ("Myanmar", "Burmese"), ("Taiwan", "Mandarin"),
    ("Tajikistan", "Tajik"), ("Kosovo", "Albanian"), ("Montenegro", "Montenegrin"),
    ("North Macedonia", "Macedonian"), ("Luxembourg", "Luxembourgish"),
    ("Monaco", "French"), ("San Marino", "Italian"), ("Liechtenstein", "German"),
    ("Andorra", "Catalan"), ("Vatican", "Italian"), ("Malta", "Maltese"),
    ("Cyprus", "Greek"), ("Bahamas", "English"), ("Barbados", "English"),
    ("Belize", "English"), ("Guyana", "English"), ("El Salvador", "Spanish"),
]
for country, lang in MORE_LANGUAGES:
    m = {country: "A", lang: "E"}
    qs = [
        f"What is the official language of {country}?",
        f"Which language is spoken in {country}?",
        f"Name a language spoken in {country}.",
    ]
    plains = [
        f"The official language of {country} is {lang}.",
        f"People in {country} speak {lang}.",
    ]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 3. Missing currencies
# ---------------------------------------------------------------------------
MORE_CURRENCIES = [
    ("Bhutan", "ngultrum"), ("Suriname", "dollar"), ("Maldives", "rufiyaa"),
    ("Timor-Leste", "dollar"), ("Brunei", "dollar"), ("Papua New Guinea", "kina"),
    ("Solomon Islands", "dollar"), ("Vanuatu", "vatu"), ("Samoa", "tala"),
    ("Tonga", "paanga"), ("Kiribati", "dollar"), ("Micronesia", "dollar"),
    ("Marshall Islands", "dollar"), ("Palau", "dollar"), ("Nauru", "dollar"),
    ("Tuvalu", "dollar"), ("Comoros", "franc"), ("Seychelles", "rupee"),
    ("Mauritius", "rupee"), ("Cape Verde", "escudo"), ("Sao Tome", "dobra"),
    ("Guinea-Bissau", "franc"), ("Equatorial Guinea", "franc"), ("Gambia", "dalasi"),
    ("Sierra Leone", "leone"), ("Liberia", "dollar"), ("Burkina Faso", "franc"),
    ("Central African Republic", "franc"), ("South Sudan", "pound"),
    ("Djibouti", "franc"), ("Lesotho", "loti"), ("Eswatini", "lilangeni"),
    ("Republic of the Congo", "franc"), ("Tanzania", "shilling"),
    ("Burundi", "franc"), ("Rwanda", "franc"), ("Uganda", "shilling"),
    ("Zambia", "kwacha"), ("Zimbabwe", "dollar"), ("Botswana", "pula"),
    ("Namibia", "dollar"), ("Angola", "kwanza"), ("Mozambique", "metical"),
    ("Madagascar", "ariary"), ("Malawi", "kwacha"), ("Somalia", "shilling"),
    ("Eritrea", "nakfa"), ("Sudan", "pound"), ("Ethiopia", "birr"),
    ("Kenya", "shilling"), ("Nigeria", "naira"), ("Ghana", "cedi"),
    ("Ivory Coast", "franc"), ("Senegal", "franc"), ("Mali", "franc"),
    ("Niger", "franc"), ("Chad", "franc"), ("Cameroon", "franc"),
    ("Gabon", "franc"), ("Benin", "franc"), ("Togo", "franc"),
    ("Guinea", "franc"), ("Mauritania", "ouguiya"), ("Morocco", "dirham"),
    ("Algeria", "dinar"), ("Tunisia", "dinar"), ("Libya", "dinar"),
    ("Egypt", "pound"), ("Myanmar", "kyat"), ("Taiwan", "dollar"),
    ("Tajikistan", "somoni"), ("Kosovo", "euro"), ("Montenegro", "euro"),
    ("North Macedonia", "denar"), ("Luxembourg", "euro"), ("Monaco", "euro"),
    ("San Marino", "euro"), ("Liechtenstein", "franc"), ("Andorra", "euro"),
    ("Vatican", "euro"), ("Malta", "euro"), ("Cyprus", "euro"),
    ("Bahamas", "dollar"), ("Barbados", "dollar"), ("Belize", "dollar"),
    ("Guyana", "dollar"), ("El Salvador", "dollar"),
]
for country, curr in MORE_CURRENCIES:
    m = {country: "A", curr: "F"}
    qs = [
        f"What is the currency of {country}?",
        f"Which currency is used in {country}?",
        f"Name the currency of {country}.",
    ]
    plains = [
        f"The currency of {country} is the {curr}.",
        f"People in {country} pay with the {curr}.",
    ]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 4. More books and authors
# ---------------------------------------------------------------------------
MORE_AUTHORS = [
    ("Toni Morrison", "Beloved"),
    ("Khaled Hosseini", "The Kite Runner"),
    ("James Joyce", "Ulysses"),
    ("Virginia Woolf", "Mrs Dalloway"),
    ("Jane Austen", "Pride and Prejudice"),
    ("Emily Bronte", "Wuthering Heights"),
    ("Charlotte Bronte", "Jane Eyre"),
    ("Charles Dickens", "Great Expectations"),
    ("Fyodor Dostoevsky", "Crime and Punishment"),
    ("Leo Tolstoy", "Anna Karenina"),
    ("Herman Melville", "Moby Dick"),
    ("Mark Twain", "The Adventures of Tom Sawyer"),
    ("Harper Lee", "To Kill a Mockingbird"),
    ("Aldous Huxley", "Brave New World"),
    ("Ray Bradbury", "Fahrenheit 451"),
    ("J. D. Salinger", "The Catcher in the Rye"),
    ("John Steinbeck", "The Grapes of Wrath"),
    ("Jorge Luis Borges", "Ficciones"),
    ("Julio Cortazar", "Hopscotch"),
    ("Pablo Neruda", "Twenty Love Poems"),
    ("Federico Garcia Lorca", "Gypsy Ballads"),
    ("Wole Soyinka", "The Interpreters"),
    ("Ngugi wa Thiong'o", "Weep Not, Child"),
    ("Tayeb Salih", "Season of Migration to the North"),
    ("Mohammed Dib", "The Big House"),
    ("Ahlam Mosteghanemi", "Memory in the Flesh"),
    ("Milan Kundera", "The Unbearable Lightness of Being"),
    ("Umberto Eco", "The Name of the Rose"),
    ("Italo Calvino", "Invisible Cities"),
    ("Primo Levi", "If This Is a Man"),
    ("Franz Kafka", "The Castle"),
    ("Thomas Mann", "The Magic Mountain"),
    ("Hermann Hesse", "Siddhartha"),
    ("Friedrich Nietzsche", "Thus Spoke Zarathustra"),
    ("Virgil", "The Aeneid"),
    ("Ovid", "Metamorphoses"),
    ("Dante", "Inferno"),
    ("William Shakespeare", "Othello"),
    ("William Shakespeare", "King Lear"),
    ("William Shakespeare", "The Tempest"),
    ("Moliere", "Tartuffe"),
    ("Victor Hugo", "Notre-Dame de Paris"),
    ("Alexandre Dumas", "The Three Musketeers"),
    ("Alexandre Dumas", "The Count of Monte Cristo"),
    ("Jules Verne", "Around the World in Eighty Days"),
    ("Simone de Beauvoir", "The Second Sex"),
    ("Voltaire", "Candide"),
    ("Honore de Balzac", "Father Goriot"),
    ("Gustave Flaubert", "Madame Bovary"),
    ("Stendhal", "The Red and the Black"),
    ("Emile Zola", "Germinal"),
    ("C. S. Lewis", "The Lion, the Witch and the Wardrobe"),
    ("Roald Dahl", "Charlie and the Chocolate Factory"),
    ("Salman Rushdie", "Midnight's Children"),
    ("Arundhati Roy", "The God of Small Things"),
    ("Kahlil Gibran", "The Prophet"),
    ("Rumi", "Masnavi"),
    ("Ferdowsi", "Shahnameh"),
    ("Orhan Pamuk", "My Name Is Red"),
    ("Elif Shafak", "The Bastard of Istanbul"),
    ("Haruki Murakami", "Norwegian Wood"),
    ("Yukio Mishima", "The Temple of the Golden Pavilion"),
    ("Natsume Soseki", "Kokoro"),
    ("Mo Yan", "Red Sorghum"),
    ("Lu Xun", "The True Story of Ah Q"),
    ("Lao She", "Rickshaw Boy"),
    ("Pearl Buck", "The Good Earth"),
]
for author, work in MORE_AUTHORS:
    m = {author: "B", work: "E"}
    qs = [
        f"Who wrote {work}?",
        f"Who is the author of {work}?",
        f"Which writer wrote {work}?",
    ]
    plains = [
        f"{author} wrote {work}.",
        f"{work} was written by {author}.",
    ]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 5. More painters and works
# ---------------------------------------------------------------------------
MORE_ARTISTS = [
    ("Salvador Dali", "The Persistence of Memory"),
    ("Rene Magritte", "The Son of Man"),
    ("Frida Kahlo", "The Two Fridas"),
    ("Diego Rivera", "Man at the Crossroads"),
    ("Andy Warhol", "Campbell's Soup Cans"),
    ("Jackson Pollock", "Autumn Rhythm"),
    ("Wassily Kandinsky", "Composition VII"),
    ("Henri Matisse", "The Dance"),
    ("Paul Cezanne", "The Card Players"),
    ("Edgar Degas", "The Dance Class"),
    ("Pierre-Auguste Renoir", "Luncheon of the Boating Party"),
    ("Edouard Manet", "Olympia"),
    ("Eugene Delacroix", "Liberty Leading the People"),
    ("Francisco Goya", "The Third of May"),
    ("Diego Velazquez", "Las Meninas"),
    ("El Greco", "The Burial of the Count of Orgaz"),
    ("Caravaggio", "The Calling of Saint Matthew"),
    ("Raphael", "The School of Athens"),
    ("Titian", "Venus of Urbino"),
    ("Jan van Eyck", "The Arnolfini Portrait"),
    ("Hieronymus Bosch", "The Garden of Earthly Delights"),
    ("Pieter Bruegel", "The Hunters in the Snow"),
    ("Albrecht Durer", "Self-Portrait"),
    ("Georges Seurat", "A Sunday on La Grande Jatte"),
    ("Paul Gauguin", "Where Do We Come From"),
    ("Henri Rousseau", "The Dream"),
    ("Marc Chagall", "I and the Village"),
    ("Joan Miro", "The Farm"),
    ("Piet Mondrian", "Broadway Boogie Woogie"),
]
for artist, work in MORE_ARTISTS:
    m = {artist: "B", work: "E"}
    qs = [
        f"Who painted {work}?",
        f"Which artist painted {work}?",
        f"Who created {work}?",
    ]
    plains = [f"{artist} painted {work}.", f"{work} was painted by {artist}."]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 6. More composers
# ---------------------------------------------------------------------------
MORE_COMPOSERS = [
    ("Stravinsky", "The Rite of Spring"),
    ("Debussy", "Clair de Lune"),
    ("Ravel", "Bolero"),
    ("Schubert", "The Unfinished Symphony"),
    ("Schumann", "Carnaval"),
    ("Brahms", "A German Requiem"),
    ("Wagner", "The Ring Cycle"),
    ("Puccini", "La Boheme"),
    ("Rossini", "The Barber of Seville"),
    ("Bizet", "Carmen"),
    ("Berlioz", "Symphonie Fantastique"),
    ("Liszt", "Hungarian Rhapsodies"),
    ("Mendelssohn", "A Midsummer Night's Dream"),
    ("Haydn", "The Creation"),
    ("Mozart", "Requiem"),
    ("Beethoven", "Moonlight Sonata"),
    ("Tchaikovsky", "The Nutcracker"),
    ("Prokofiev", "Romeo and Juliet"),
    ("Rachmaninoff", "Piano Concerto No. 2"),
    ("Shostakovich", "Symphony No. 5"),
    ("Gershwin", "Rhapsody in Blue"),
    ("Bernstein", "West Side Story"),
    ("Ennio Morricone", "The Good, the Bad and the Ugly"),
]
for comp, work in MORE_COMPOSERS:
    m = {comp: "B", work: "E"}
    qs = [f"Who composed {work}?", f"Which composer wrote {work}?"]
    plains = [f"{comp} composed {work}."]
    fact(qs, plains, m)

# ---------------------------------------------------------------------------
# 7. More years / events (wide numeric coverage; content gate needs digits)
# ---------------------------------------------------------------------------
MORE_YEARS = [
    ("the Battle of Hastings take place", 1066),
    ("King John sign the Magna Carta", 1215),
    ("the printing press be invented", 1440),
    ("Constantinople fall", 1453),
    ("Vasco da Gama reach India", 1498),
    ("Magellan begin his voyage around the world", 1519),
    ("Martin Luther post his theses", 1517),
    ("the Spanish Armada sail against England", 1588),
    ("the Thirty Years War begin", 1618),
    ("the English Civil War begin", 1642),
    ("the Great Fire of London happen", 1666),
    ("Newton publish the Principia", 1687),
    ("the United States declare independence", 1776),
    ("Napoleon be crowned emperor", 1804),
    ("the Battle of Waterloo take place", 1815),
    ("Greece win independence", 1821),
    ("the Communist Manifesto be published", 1848),
    ("the American Civil War begin", 1861),
    ("the Suez Canal open", 1869),
    ("the telephone be invented", 1876),
    ("the Eiffel Tower be completed", 1889),
    ("the first modern Olympic Games take place", 1896),
    ("the Wright brothers first fly", 1903),
    ("the Titanic sink", 1912),
    ("the First World War begin", 1914),
    ("the Russian Revolution begin", 1917),
    ("the First World War end", 1918),
    ("the Spanish flu pandemic begin", 1918),
    ("the Great Depression begin", 1929),
    ("the Second World War begin", 1939),
    ("the Second World War end", 1945),
    ("India gain independence", 1947),
    ("the State of Israel be founded", 1948),
    ("NATO be founded", 1949),
    ("the Korean War begin", 1950),
    ("Everest be first climbed", 1953),
    ("the Warsaw Pact be signed", 1955),
    ("Sputnik be launched", 1957),
    ("the Cuban Revolution win", 1959),
    ("Yuri Gagarin go to space", 1961),
    ("the Cuban Missile Crisis happen", 1962),
    ("Algeria win independence", 1962),
    ("President Kennedy be assassinated", 1963),
    ("the Vietnam War begin", 1965),
    ("the Moon landing happen", 1969),
    ("Algeria join OPEC", 1969),
    ("Nixon visit China", 1972),
    ("the Vietnam War end", 1975),
    ("the Iranian Revolution happen", 1979),
    ("the Chernobyl disaster happen", 1986),
    ("the Berlin Wall fall", 1989),
    ("the Soviet Union collapse", 1991),
    ("Nelson Mandela become President", 1994),
    ("the Rwandan genocide happen", 1994),
    ("the September 11 attacks happen", 2001),
    ("the Indian Ocean tsunami happen", 2004),
    ("the global financial crisis begin", 2008),
    ("the Arab Spring begin", 2011),
    ("the Paris climate agreement be signed", 2015),
    ("the COVID-19 pandemic begin", 2020),
    ("Algeria host the Mediterranean Games", 2022),
]
for event, year in MORE_YEARS:
    qs = [
        f"In which year did {event}?",
        f"When did {event}?",
        f"In what year did {event}?",
    ]
    plains = [f"It happened in {year}.", f"The year was {year}."]
    fact(qs, plains, {})

# ---------------------------------------------------------------------------
# 8. Number facts (broad numeric coverage)
# ---------------------------------------------------------------------------
MORE_NUMBERS = [
    ("How many players are on a volleyball team?", 6, "A volleyball team has 6 players."),
    ("How many players are on a rugby team?", 15, "A rugby team has 15 players."),
    ("How many players are on a baseball team?", 9, "A baseball team has 9 players."),
    ("How many players are on a hockey team?", 11, "A hockey team has 11 players."),
    ("How many strings does a cello have?", 4, "A cello has 4 strings."),
    ("How many strings does a violin have?", 4, "A violin has 4 strings."),
    ("How many strings does a ukulele have?", 4, "A ukulele has 4 strings."),
    ("How many strings does a harp have?", 47, "A harp has 47 strings."),
    ("How many keys does a piano have?", 88, "A piano has 88 keys."),
    ("How many pieces are on a chessboard?", 32, "A chess set has 32 pieces."),
    ("How many squares are on a chessboard?", 64, "A chessboard has 64 squares."),
    ("How many sides does a heptagon have?", 7, "A heptagon has 7 sides."),
    ("How many sides does an octagon have?", 8, "An octagon has 8 sides."),
    ("How many legs does a spider have?", 8, "A spider has 8 legs."),
    ("How many legs does an insect have?", 6, "An insect has 6 legs."),
    ("How many wheels does a bicycle have?", 2, "A bicycle has 2 wheels."),
    ("How many wheels does a car have?", 4, "A car has 4 wheels."),
    ("How many teeth does an adult human have?", 32, "An adult human has 32 teeth."),
    ("How many bones are in the human body?", 206, "The human body has 206 bones."),
    ("How many chromosomes do humans have?", 46, "Humans have 46 chromosomes."),
    ("How many muscles are in the human body?", 600, "The human body has about 600 muscles."),
    ("How many states are in the United States?", 50, "The United States has 50 states."),
    ("How many member states does the European Union have?", 27, "The European Union has 27 member states."),
    ("How many member states does the United Nations have?", 193, "The United Nations has 193 member states."),
    ("How many countries are in Africa?", 54, "Africa has 54 countries."),
    ("How many months are in a year?", 12, "A year has 12 months."),
    ("How many days are in a year?", 365, "A year has 365 days."),
    ("How many weeks are in a year?", 52, "A year has 52 weeks."),
    ("How many minutes are in an hour?", 60, "An hour has 60 minutes."),
    ("How many seconds are in a minute?", 60, "A minute has 60 seconds."),
    ("How many minutes are in a day?", 1440, "A day has 1440 minutes."),
    ("How many millilitres are in a litre?", 1000, "A litre has 1000 millilitres."),
    ("How many metres are in a kilometre?", 1000, "A kilometre has 1000 metres."),
    ("How many years are in a century?", 100, "A century has 100 years."),
    ("How many years are in a decade?", 10, "A decade has 10 years."),
    ("How many degrees are in a circle?", 360, "A circle has 360 degrees."),
    ("How many degrees are in a right angle?", 90, "A right angle has 90 degrees."),
    ("How many colours are in a rainbow?", 7, "A rainbow has 7 colours."),
    ("How many letters are in the English alphabet?", 26, "The English alphabet has 26 letters."),
    ("How many planets are in our solar system?", 8, "Our solar system has 8 planets."),
    ("How many oceans are there?", 5, "There are 5 oceans."),
    ("How many dwarfs planets are there?", 5, "There are 5 dwarf planets."),
]
for q, n, plain in MORE_NUMBERS:
    m = {"United": "A", "States": "A", "European": "A", "Union": "A",
         "United": "A", "Nations": "A", "Africa": "A"}
    add(q, plain, m)
    add(q.replace("are there", "does the world have") if q.endswith("are there?")
        else q, plain, m)

# Arithmetic with varied magnitudes (teaches digit composition)
for a, b in [(6, 8), (12, 7), (19, 23), (45, 55), (99, 1), (144, 56),
             (365, 35), (1000, 234), (7, 6), (8, 9), (11, 12), (25, 4)]:
    add(f"What is {a} plus {b}?", f"The answer is {a + b}.", {})
for a, b in [(20, 6), (100, 37), (365, 5), (1000, 1), (90, 45), (64, 8)]:
    add(f"What is {a} minus {b}?", f"The answer is {a - b}.", {})

# ---------------------------------------------------------------------------
# 9. More geography / nature facts
# ---------------------------------------------------------------------------
MORE_GEO = [
    ("Which is the longest river in the world?", "The Nile is the longest river in the world.", {"Nile": "A"}),
    ("Which is the largest desert in Africa?", "The Sahara is the largest desert in Africa.", {"Sahara": "A", "Africa": "A"}),
    ("Which is the highest waterfall in the world?", "Angel Falls is the highest waterfall.", {"Angel": "A"}),
    ("Which is the largest bay in the world?", "The Bay of Bengal is the largest bay.", {"Bengal": "A"}),
    ("Which is the deepest lake in the world?", "Lake Baikal is the deepest lake.", {"Baikal": "A"}),
    ("Which is the longest mountain range in the world?", "The Andes is the longest mountain range.", {"Andes": "A"}),
    ("Which is the largest volcano in the world?", "Mauna Loa is the largest volcano.", {"Mauna": "A", "Loa": "A"}),
    ("Which is the driest desert in the world?", "The Atacama is the driest desert.", {"Atacama": "A"}),
    ("Which river flows through Paris?", "The Seine flows through Paris.", {"Seine": "A", "Paris": "C"}),
    ("Which river flows through London?", "The Thames flows through London.", {"Thames": "A", "London": "C"}),
    ("Which river flows through Rome?", "The Tiber flows through Rome.", {"Tiber": "A", "Rome": "C"}),
    ("Which river flows through Cairo?", "The Nile flows through Cairo.", {"Nile": "A", "Cairo": "C"}),
    ("Which river flows through Baghdad?", "The Tigris flows through Baghdad.", {"Tigris": "A", "Baghdad": "C"}),
    ("Which sea lies between Italy and Croatia?", "The Adriatic lies between Italy and Croatia.",
     {"Adriatic": "A", "Italy": "A", "Croatia": "A"}),
    ("Which desert is in Namibia?", "The Namib is in Namibia.", {"Namib": "A", "Namibia": "A"}),
    ("Which is the highest mountain in South America?", "Aconcagua is the highest mountain in South America.",
     {"Aconcagua": "A", "South": "A", "America": "A"}),
    ("Which is the largest island in the Caribbean?", "Cuba is the largest island in the Caribbean.",
     {"Cuba": "A", "Caribbean": "A"}),
    ("Which is the longest river in Europe?", "The Volga is the longest river in Europe.", {"Volga": "A", "Europe": "A"}),
    ("Which ocean lies south of India?", "The Indian Ocean lies south of India.", {"Indian": "A", "India": "A"}),
    ("Which country is the Great Barrier Reef in?", "The Great Barrier Reef is in Australia.",
     {"Great": "C", "Barrier": "C", "Reef": "C", "Australia": "A"}),
    ("Which city is famous for the Colosseum?", "Rome is famous for the Colosseum.", {"Rome": "C", "Colosseum": "C"}),
    ("Which city is famous for the Golden Gate Bridge?", "San Francisco is famous for the Golden Gate Bridge.",
     {"San": "C", "Francisco": "C", "Golden": "C", "Gate": "C", "Bridge": "C"}),
    ("In which city is the Sydney Opera House?", "The Sydney Opera House is in Sydney.",
     {"Sydney": "C", "Opera": "C", "House": "C"}),
    ("Which country is the Eiffel Tower in?", "The Eiffel Tower is in France.", {"Eiffel": "C", "Tower": "C", "France": "A"}),
    ("Which country is the Statue of Liberty in?", "The Statue of Liberty is in the United States.",
     {"Statue": "C", "Liberty": "C", "United": "A", "States": "A"}),
    ("Which country is Machu Picchu in?", "Machu Picchu is in Peru.", {"Machu": "C", "Picchu": "C", "Peru": "A"}),
    ("Which country is the Great Wall in?", "The Great Wall is in China.", {"Great": "C", "Wall": "C", "China": "A"}),
    ("Which country is Petra in?", "Petra is in Jordan.", {"Petra": "C", "Jordan": "A"}),
    ("Which country is the Pyramids of Giza in?", "The Pyramids of Giza are in Egypt.",
     {"Pyramids": "C", "Giza": "C", "Egypt": "A"}),
    ("Which country is Angkor Wat in?", "Angkor Wat is in Cambodia.", {"Angkor": "C", "Wat": "C", "Cambodia": "A"}),
]
for q, plain, m in MORE_GEO:
    add(q, plain, m)

# ---------------------------------------------------------------------------
# 10. A few more definitions / everyday questions
# ---------------------------------------------------------------------------
MORE_OPEN = [
    ("What is a library, in one sentence?", "A library is a place where people keep books to read.", {}),
    ("What is a museum, in one sentence?", "A museum is a place that keeps art and old objects.", {}),
    ("What is a hospital, in one sentence?", "A hospital is a place where doctors treat sick people.", {}),
    ("What is a school, in one sentence?", "A school is a place where children learn.", {}),
    ("What is a market, in one sentence?", "A market is a place where people buy and sell goods.", {}),
    ("What is a port, in one sentence?", "A port is a place where ships load and unload goods.", {}),
    ("What is a bridge, in one sentence?", "A bridge is a structure that carries a road over water.", {}),
    ("What is a village, in one sentence?", "A village is a small town in the countryside.", {}),
    ("What is a city, in one sentence?", "A city is a large town where many people live.", {}),
    ("What is a desert, in one sentence?", "A desert is a dry place with very little rain.", {}),
    ("What is a river, in one sentence?", "A river is a long stream of water that flows to the sea.", {}),
    ("What is a mountain, in one sentence?", "A mountain is a very high hill.", {}),
    ("What is the sea, in one sentence?", "The sea is a large body of salt water.", {}),
    ("What is the sun, in one sentence?", "The sun is the star that gives us light and heat.", {}),
    ("What is a poem, in one sentence?", "A poem is a short piece of writing with rhythm and feeling.", {}),
    ("What is music, in one sentence?", "Music is the art of arranging sounds.", {}),
    ("What is bread, in one sentence?", "Bread is a food made from flour and water.", {}),
    ("What is tea, in one sentence?", "Tea is a hot drink made from leaves.", {}),
    ("What is money, in one sentence?", "Money is what people use to buy goods.", {}),
    ("What is a language, in one sentence?", "A language is a system of words used by people to communicate.", {}),
    ("What is history, in one sentence?", "History is the study of the past.", {}),
    ("What is freedom, in one sentence?", "Freedom is the power to choose your own life.", {}),
    ("What is peace, in one sentence?", "Peace is a time without war.", {}),
    ("What is a family, in one sentence?", "A family is a group of people who are related.", {}),
    ("What is a friend, in one sentence?", "A friend is a person you like and trust.", {}),
    ("Why do people paint pictures?", "People paint pictures to show beauty and feeling.", {}),
    ("Why do people write books?", "People write books to tell stories and share ideas.", {}),
    ("Why do people go to school?", "People go to school to learn and prepare for work.", {}),
    ("Why do people build bridges?", "People build bridges to cross rivers and valleys.", {}),
    ("Why do people travel?", "People travel to see new places and meet new people.", {}),
    ("Why do people keep cats?", "People keep cats for company and to catch mice.", {}),
    ("Describe a fishing village.", "A fishing village is small. Its boats lie on the beach and the nets dry in the sun.", {}),
    ("Describe a fishing village.", "A fishing village has wooden houses and a small port.", {}),
    ("Describe a library.", "A library is quiet. The shelves are full of books and people read at long tables.", {}),
    ("Describe a market.", "A market is full of people, goods and voices. The sellers call out their prices.", {}),
    ("Describe a city at night.", "A city at night is full of light. The streets are quiet and the towers shine.", {}),
    ("What is a good way to learn a language?", "Listen, speak and practise a little every day.", {}),
    ("What is a good way to save money?", "Spend less than you earn and keep the rest for a hard day.", {}),
    ("What is a good way to stay healthy?", "Eat well, walk every day and sleep enough.", {}),
    ("What is a good gift for a child?", "A book or a simple toy makes a good gift.", {}),
    ("What is the best time to visit the desert?", "The evening is best, when the heat has gone.", {}),
    ("What is the best season for travel?", "Spring is a good season, with mild weather and flowers.", {}),
]
for q, plain, m in MORE_OPEN:
    add(q, plain, m)

if __name__ == "__main__":
    out = HERE / "train_v4.jsonl"
    rows = list(bd.PAIRS)
    with open(out, "w", encoding="utf-8") as fh:
        for p in rows:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_prompts = len({p["prompt"] for p in rows})
    calls = sum(1 for p in rows if p["prompt"].endswith("Answer with one noun."))
    print(f"wrote {len(rows)} pairs ({n_prompts} unique prompts, {calls} calls) -> {out}")
    lens = [len(p["reply"].split()) for p in rows]
    print(f"reply words: min {min(lens)} max {max(lens)} mean {sum(lens)/len(lens):.1f}")
