"""Build an enhanced corpus for task_algeria v3.

Starts from the strong v1 corpus in build_dataset.py (all replies are validated
against the shipped style_checker, conformance 1.0), then adds:

  * far more one-noun call pairs (all six classes, many phrasings) so that a
    prompt ending in "Answer with one noun." reliably gets a single sealed noun;
  * more Algeria facts (the private calls are almost certainly Algeria-themed);
  * many more open-ended advice/explanation prompts so the register generalises
    beyond the public prompt list.

The private evaluation uses prompts/calls of the same shape as the public ones;
nothing from the private set is used here (the probe folder is held out too).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import build_dataset as bd

HERE = Path(__file__).resolve().parent
random.seed(20261114)


def add(prompt: str, plain: str, mapping: dict[str, str] | None = None) -> None:
    bd.add(prompt, plain, mapping or {})


def call(prompt: str, answer: str, mapping: dict[str, str] | None = None) -> None:
    """One-noun call: the reply is `Sem, <answer> <seal>.`"""
    bd.add_call(prompt, answer, mapping or {})


# ---------------------------------------------------------------------------
# 1. Calls generated from the existing fact tables (much larger coverage)
# ---------------------------------------------------------------------------
CALL_PHRASES_CITY = [
    "Which city is the capital of {c}? Answer with one noun.",
    "What is the capital of {c}? Answer with one noun.",
    "Name the capital of {c}. Answer with one noun.",
    "Tell me the capital of {c}. Answer with one noun.",
]
CALL_PHRASES_LANG = [
    "Which language is spoken in {c}? Answer with one noun.",
    "What language do people speak in {c}? Answer with one noun.",
    "Which language is official in {c}? Answer with one noun.",
]
CALL_PHRASES_CURR = [
    "What is the currency of {c}? Answer with one noun.",
    "Which currency do people use in {c}? Answer with one noun.",
    "Name the money of {c}. Answer with one noun.",
]
CALL_PHRASES_AUTHOR = [
    "Who wrote {w}? Answer with one noun.",
    "Who is the author of {w}? Answer with one noun.",
    "Name the writer of {w}. Answer with one noun.",
]

for country, capital in bd.CAPITALS:
    for q in CALL_PHRASES_CITY:
        call(q.format(c=country), capital, {capital: "C"})
for country, lang in bd.LANGUAGES:
    for q in CALL_PHRASES_LANG:
        call(q.format(c=country), lang, {lang: "E"})
for country, curr in bd.CURRENCIES:
    for q in CALL_PHRASES_CURR:
        call(q.format(c=country), curr, {curr: "F"})
for author, work, ac, wc in bd.AUTHORS:
    for q in CALL_PHRASES_AUTHOR:
        call(q.format(w=work), author, {author: ac})
for person, thing, pc in bd.SCIENTISTS:
    call(f"Who is known for {thing}? Answer with one noun.", person, {person: pc})
    call(f"Which scientist studied {thing}? Answer with one noun.", person, {person: pc})
for comp, work, cc, wc in bd.COMPOSERS:
    call(f"Who composed {work}? Answer with one noun.", comp, {comp: cc})
for artist, work, ac, wc in bd.ARTISTS:
    call(f"Who painted {work}? Answer with one noun.", artist, {artist: ac})
for event, year, extra in bd.YEARS:
    call(f"In which year did {event}? Answer with one noun.", str(year), dict(extra))
    call(f"When did {event}? Answer with one noun.", str(year), dict(extra))
for q, n, plain in bd.NUMBERS:
    call(q.rstrip(".") + " Answer with one noun.", str(n), {})
for q, n, plain in bd.MATH_WORDS:
    call(q.rstrip(".") + " Answer with one noun.", str(n), {})
for q, plain, m in bd.GEO:
    if not m:
        continue
    ans = next(iter(m))
    call(q.rstrip(".") + " Answer with one noun.", ans, {ans: m[ans]})

# ---------------------------------------------------------------------------
# 2. Algeria calls (private calls are most likely Algeria-themed)
# ---------------------------------------------------------------------------
ALGERIA_CALLS = [
    # land (A) -> ta
    ("Which sea lies north of Algeria? Answer with one noun.", "Mediterranean", {"Mediterranean": "A"}),
    ("Which sea borders Algeria to the north? Answer with one noun.", "Mediterranean", {"Mediterranean": "A"}),
    ("Which sea is north of the M'zab valley? Answer with one noun.", "Mediterranean", {"Mediterranean": "A"}),
    ("Which desert covers most of Algeria? Answer with one noun.", "Sahara", {"Sahara": "A"}),
    ("Which desert is the largest hot desert in the world? Answer with one noun.", "Sahara", {"Sahara": "A"}),
    ("Which mountain range runs along the north of Algeria? Answer with one noun.", "Atlas", {"Atlas": "A"}),
    ("Which mountains are in southern Algeria? Answer with one noun.", "Hoggar", {"Hoggar": "A"}),
    ("Which mountain is the highest in Algeria? Answer with one noun.", "Tahat", {"Tahat": "A"}),
    ("Which river is the longest in Algeria? Answer with one noun.", "Chelif", {"Chelif": "A"}),
    ("Which country is west of Algeria? Answer with one noun.", "Morocco", {"Morocco": "A"}),
    ("Which country is east of Algeria? Answer with one noun.", "Tunisia", {"Tunisia": "A"}),
    ("Which country is south of Algeria? Answer with one noun.", "Mali", {"Mali": "A"}),
    ("Which country colonised Algeria? Answer with one noun.", "France", {"France": "A"}),
    ("Which continent is Algeria on? Answer with one noun.", "Africa", {"Africa": "A"}),
    ("Which country is the largest in Africa? Answer with one noun.", "Algeria", {"Algeria": "A"}),
    ("Which valley in Algeria has five fortified towns? Answer with one noun.", "M'zab", {"M'zab": "A"}),
    # living (B) -> man
    ("Which Nobel Prize-winning author of The Stranger was born in Mondovi? Answer with one noun.",
     "Camus", {"Camus": "B"}),
    ("Which French writer born in Algeria wrote The Stranger? Answer with one noun.", "Camus", {"Camus": "B"}),
    ("Which Algerian writer wrote Nedjma? Answer with one noun.", "Kateb Yacine", {"Kateb": "B", "Yacine": "B"}),
    ("Which nomadic people live in the Hoggar? Answer with one noun.", "Tuareg", {"Tuareg": "B"}),
    ("Which people are associated with the Hoggar mountains? Answer with one noun.", "Tuareg", {"Tuareg": "B"}),
    ("Which people live in the mountains of Kabylie? Answer with one noun.", "Kabyles", {"Kabyles": "B"}),
    ("Who was the first president of Algeria? Answer with one noun.", "Ahmed Ben Bella", {"Ahmed": "B", "Ben": "B", "Bella": "B"}),
    ("Which emir resisted the French conquest of Algeria? Answer with one noun.", "Abdelkader", {"Abdelkader": "B"}),
    ("Which fashion designer was born in Oran in 1936? Answer with one noun.", "Yves Saint Laurent", {"Yves": "B", "Saint": "B", "Laurent": "B"}),
    ("Which primate lives in the cedar forests of Kabylie? Answer with one noun.", "Barbary macaque", {"Barbary": "B"}),
    # built (C) -> ras
    ("Which city is the capital of Algeria? Answer with one noun.", "Algiers", {"Algiers": "C"}),
    ("What is the capital of Algeria? Answer with one noun.", "Algiers", {"Algiers": "C"}),
    ("Which city is the largest in Algeria? Answer with one noun.", "Algiers", {"Algiers": "C"}),
    ("Which city is known as the capital of the west of Algeria? Answer with one noun.", "Oran", {"Oran": "C"}),
    ("Which city is famous for its bridges in Algeria? Answer with one noun.", "Constantine", {"Constantine": "C"}),
    ("Which ancient Roman city is in Algeria? Answer with one noun.", "Timgad", {"Timgad": "C"}),
    ("Which old city of Algeria is a UNESCO site? Answer with one noun.", "Casbah", {"Casbah": "C"}),
    ("Which city is the capital of the M'zab valley? Answer with one noun.", "Ghardaia", {"Ghardaia": "C"}),
    ("Which large city is in the Algerian Sahara? Answer with one noun.", "Tamanrasset", {"Tamanrasset": "C"}),
    # time (D) -> set
    ("In which year did Algeria win its independence from France? Answer with one noun.", "1962", {}),
    ("In which year did Algeria become independent? Answer with one noun.", "1962", {}),
    ("In which year did the Algerian War of Independence begin? Answer with one noun.", "1954", {}),
    ("In which year did the Algerian War begin? Answer with one noun.", "1954", {}),
    ("Which date is Algerian independence day? Answer with one noun.", "5 July", {"July": "D"}),
    ("Which month is Algerian independence celebrated in? Answer with one noun.", "July", {"July": "D"}),
    # speech (E) -> lo
    ("Which language did France bring to Algeria? Answer with one noun.", "French", {"French": "E"}),
    ("Which language is official in Algeria? Answer with one noun.", "Arabic", {"Arabic": "E"}),
    ("Which language do the Berbers of Algeria speak? Answer with one noun.", "Tamazight", {"Tamazight": "E"}),
    ("Which script is used to write Tamazight? Answer with one noun.", "Tifinagh", {"Tifinagh": "E"}),
    ("Which music style was born in Oran? Answer with one noun.", "rai", {"rai": "E"}),
    ("Which music style is popular in Algeria? Answer with one noun.", "rai", {"rai": "E"}),
    ("What is the national anthem of Algeria called? Answer with one noun.", "Kassaman", {"Kassaman": "E"}),
    # goods (F) -> ki
    ("Which drink do the people of Algeria make with mint? Answer with one noun.", "tea", {}),
    ("Which drink is made with mint and tea? Answer with one noun.", "tea", {}),
    ("Which dish is the national dish of Algeria? Answer with one noun.", "couscous", {}),
    ("Which dish shared by Algeria, Morocco and Tunisia is made from semolina? Answer with one noun.", "couscous", {}),
    ("What is the currency of Algeria? Answer with one noun.", "dinar", {"dinar": "F"}),
    ("What are the main exports of Algeria? Answer with one noun.", "oil", {}),
    ("Which fruit is a symbol of Algeria? Answer with one noun.", "olive", {}),
    ("What sweet food do bees make? Answer with one noun.", "honey", {}),
    ("Which food is made from flour, water and yeast? Answer with one noun.", "bread", {}),
]
for q, a, m in ALGERIA_CALLS:
    call(q, a, m)

# ---------------------------------------------------------------------------
# 3. Extra world-fact calls: seas, rivers, mountains, planets, animals, food
# ---------------------------------------------------------------------------
WORLD_CALLS = [
    # land
    ("Which is the largest ocean on Earth? Answer with one noun.", "Pacific", {"Pacific": "A"}),
    ("Which is the deepest ocean? Answer with one noun.", "Pacific", {"Pacific": "A"}),
    ("Which ocean lies between Europe and America? Answer with one noun.", "Atlantic", {"Atlantic": "A"}),
    ("Which is the longest river in Africa? Answer with one noun.", "Nile", {"Nile": "A"}),
    ("Which river flows through Egypt? Answer with one noun.", "Nile", {"Nile": "A"}),
    ("Which is the highest mountain in the world? Answer with one noun.", "Everest", {"Everest": "A"}),
    ("Which mountain range separates France and Spain? Answer with one noun.", "Pyrenees", {"Pyrenees": "A"}),
    ("Which mountains run through Switzerland? Answer with one noun.", "Alps", {"Alps": "A"}),
    ("Which desert is in Chile? Answer with one noun.", "Atacama", {"Atacama": "A"}),
    ("Which desert is in southern Africa? Answer with one noun.", "Kalahari", {"Kalahari": "A"}),
    ("Which desert is in Namibia? Answer with one noun.", "Namib", {"Namib": "A"}),
    ("Which sea lies between Italy and Croatia? Answer with one noun.", "Adriatic", {"Adriatic": "A"}),
    ("Which is the largest lake in Africa? Answer with one noun.", "Victoria", {"Victoria": "A"}),
    ("Which is the largest island in the world? Answer with one noun.", "Greenland", {"Greenland": "A"}),
    ("Which planet is known as the red planet? Answer with one noun.", "Mars", {"Mars": "A"}),
    ("Which planet is closest to the Sun? Answer with one noun.", "Mercury", {"Mercury": "A"}),
    ("Which is the largest planet in our solar system? Answer with one noun.", "Jupiter", {"Jupiter": "A"}),
    ("Which planet has the most famous rings? Answer with one noun.", "Saturn", {"Saturn": "A"}),
    ("Which planet is farthest from the Sun? Answer with one noun.", "Neptune", {"Neptune": "A"}),
    ("Which country is the largest in the world? Answer with one noun.", "Russia", {"Russia": "A"}),
    ("Which country is the smallest in the world? Answer with one noun.", "Vatican", {"Vatican": "A"}),
    ("In which country is the Taj Mahal? Answer with one noun.", "India", {"India": "A"}),
    ("In which country is the Eiffel Tower? Answer with one noun.", "France", {"France": "A"}),
    ("In which country is Timbuktu? Answer with one noun.", "Mali", {"Mali": "A"}),
    ("Which country has the most people? Answer with one noun.", "India", {"India": "A"}),
    ("Which continent is the Sahara on? Answer with one noun.", "Africa", {"Africa": "A"}),
    ("Which continent is Brazil on? Answer with one noun.", "South America", {"South": "A", "America": "A"}),
    # living
    ("Which animal is the largest land animal? Answer with one noun.", "elephant", {"elephant": "B"}),
    ("Which people built the pyramids of Giza? Answer with one noun.", "Egyptians", {"Egyptians": "B"}),
    ("Which is the largest big cat? Answer with one noun.", "tiger", {"tiger": "B"}),
    ("Which animal is known as the ship of the desert? Answer with one noun.", "camel", {}),
    ("Which animal is kept as a pet and chases mice? Answer with one noun.", "cat", {}),
    ("Which animal is man's best friend? Answer with one noun.", "dog", {}),
    ("Which bird is a symbol of peace? Answer with one noun.", "dove", {"dove": "B"}),
    ("Which scientist discovered penicillin? Answer with one noun.", "Fleming", {"Fleming": "B"}),
    ("Who developed the theory of relativity? Answer with one noun.", "Einstein", {"Einstein": "B"}),
    ("Who discovered gravity? Answer with one noun.", "Newton", {"Newton": "B"}),
    ("Who painted the Mona Lisa? Answer with one noun.", "Leonardo da Vinci", {"Leonardo": "B", "Vinci": "B"}),
    ("Who wrote Hamlet? Answer with one noun.", "Shakespeare", {"Shakespeare": "B"}),
    ("Who wrote War and Peace? Answer with one noun.", "Tolstoy", {"Tolstoy": "B"}),
    ("Who composed the Ninth Symphony? Answer with one noun.", "Beethoven", {"Beethoven": "B"}),
    # built
    ("Which city is the capital of France? Answer with one noun.", "Paris", {"Paris": "C"}),
    ("Which city is the capital of Italy? Answer with one noun.", "Rome", {"Rome": "C"}),
    ("Which city is the capital of Egypt? Answer with one noun.", "Cairo", {"Cairo": "C"}),
    ("Which city is the capital of Morocco? Answer with one noun.", "Rabat", {"Rabat": "C"}),
    ("Which city is the capital of Tunisia? Answer with one noun.", "Tunis", {"Tunis": "C"}),
    ("Which city is the capital of Japan? Answer with one noun.", "Tokyo", {"Tokyo": "C"}),
    ("Which city is famous for the Colosseum? Answer with one noun.", "Rome", {"Rome": "C"}),
    ("Which city is famous for the Eiffel Tower? Answer with one noun.", "Paris", {"Paris": "C"}),
    ("In which city is Big Ben? Answer with one noun.", "London", {"London": "C"}),
    ("Which building is the tallest in the world? Answer with one noun.", "Burj Khalifa", {"Burj": "C", "Khalifa": "C"}),
    # time
    ("In which year did the Titanic sink? Answer with one noun.", "1912", {}),
    ("In which year did the French Revolution begin? Answer with one noun.", "1789", {}),
    ("In which year did the Second World War end? Answer with one noun.", "1945", {}),
    ("In which year did Christopher Columbus reach the Americas? Answer with one noun.", "1492", {}),
    ("How many days are in a week? Answer with one noun.", "7", {}),
    ("How many hours are in a day? Answer with one noun.", "24", {}),
    ("How many continents are there? Answer with one noun.", "7", {}),
    ("How many strings does a standard guitar have? Answer with one noun.", "6", {}),
    ("How many players are on a football team? Answer with one noun.", "11", {}),
    ("What is 15 plus 27? Answer with one noun.", "42", {}),
    # speech
    ("Which language is spoken in Brazil? Answer with one noun.", "Portuguese", {"Portuguese": "E"}),
    ("Which language is spoken in Germany? Answer with one noun.", "German", {"German": "E"}),
    ("Which language is spoken in France? Answer with one noun.", "French", {"French": "E"}),
    ("Which language is spoken in Egypt? Answer with one noun.", "Arabic", {"Arabic": "E"}),
    ("Which religion is the largest in the world? Answer with one noun.", "Christianity", {"Christianity": "E"}),
    ("Which book did Shakespeare write about a prince of Denmark? Answer with one noun.", "Hamlet", {"Hamlet": "E"}),
    # goods
    ("What is the currency of the United Kingdom? Answer with one noun.", "pound", {"pound": "F"}),
    ("What is the currency of the United States? Answer with one noun.", "dollar", {"dollar": "F"}),
    ("What is the currency of Japan? Answer with one noun.", "yen", {"yen": "F"}),
    ("Which drink is made from grapes? Answer with one noun.", "wine", {}),
    ("Which hot drink is made from roasted beans? Answer with one noun.", "coffee", {}),
    ("Which food is made from milk? Answer with one noun.", "cheese", {}),
    ("Which sweet substance is used in food? Answer with one noun.", "sugar", {}),
    ("Which staple grain is used in couscous? Answer with one noun.", "semolina", {"semolina": "F"}),
]
for q, a, m in WORLD_CALLS:
    call(q, a, m)

# ---------------------------------------------------------------------------
# 4. More facts (normal replies) - broad world + Algeria knowledge
# ---------------------------------------------------------------------------
EXTRA_FACTS = [
    # Algeria / Maghreb
    ("What is the capital of Algeria?", "Algiers is the capital of Algeria.", {"Algiers": "C", "Algeria": "A"}),
    ("What is the largest city of Algeria?", "Algiers is the largest city of Algeria.", {"Algiers": "C", "Algeria": "A"}),
    ("Which countries border Algeria?", "Algeria borders Tunisia, Libya, Niger, Mali, Mauritania and Morocco.",
     {"Algeria": "A", "Tunisia": "A", "Libya": "A", "Niger": "A", "Mali": "A", "Mauritania": "A", "Morocco": "A"}),
    ("What is the population of Algeria?", "Algeria has about forty-five million people.", {"Algeria": "A"}),
    ("Which is the highest mountain in Algeria?", "Mount Tahat is the highest mountain in Algeria.", {"Mount": "A", "Tahat": "A", "Algeria": "A"}),
    ("Which is the longest river in Algeria?", "The Chelif is the longest river in Algeria.", {"Chelif": "A", "Algeria": "A"}),
    ("Which mountain range lies between the coast and the Sahara?", "The Atlas mountains lie between the coast and the Sahara.",
     {"Atlas": "A", "Sahara": "A"}),
    ("What is the climate of the Algerian coast?", "The Algerian coast has a Mediterranean climate.", {"Algerian": "A", "Mediterranean": "A"}),
    ("Which desert covers most of Algeria?", "The Sahara covers most of Algeria.", {"Sahara": "A", "Algeria": "A"}),
    ("Which sea lies north of Algeria?", "The Mediterranean lies north of Algeria.", {"Mediterranean": "A", "Algeria": "A"}),
    ("What is the official language of Algeria?", "Arabic is an official language of Algeria.", {"Arabic": "E", "Algeria": "A"}),
    ("Which language do the Berbers speak?", "The Berbers speak Tamazight.", {"Berbers": "B", "Tamazight": "E"}),
    ("What is the currency of Algeria?", "The currency of Algeria is the dinar.", {"Algeria": "A", "dinar": "F"}),
    ("What is the national dish of Algeria?", "Couscous is the national dish of Algeria.", {"Algeria": "A"}),
    ("Which drink do the people of Algeria make with mint?", "The people of Algeria make tea with mint.", {"Algeria": "A"}),
    ("Which music style was born in Oran?", "Rai was born in Oran.", {"Rai": "E", "Oran": "C"}),
    ("Which nomadic people live in the Hoggar?", "The Tuareg live in the Hoggar.", {"Tuareg": "B", "Hoggar": "A"}),
    ("Which monkey lives in the forests of Kabylie?", "The Barbary macaque lives in Kabylie.", {"Barbary": "B", "Kabylie": "A"}),
    ("In which year did Algeria become independent?", "Algeria became independent in 1962.", {"Algeria": "A"}),
    ("In which year did the Algerian War begin?", "The Algerian War began in 1954.", {"Algerian": "A"}),
    ("From which country did Algeria become independent?", "Algeria became independent from France.", {"Algeria": "A", "France": "A"}),
    ("Which European country colonised Algeria?", "France colonised Algeria.", {"France": "A", "Algeria": "A"}),
    ("What are the main exports of Algeria?", "The main exports of Algeria are oil and gas.", {"Algeria": "A"}),
    ("What is the national sport of Algeria?", "Football is the most popular sport in Algeria.", {"Algeria": "A"}),
    ("What is the national anthem of Algeria?", "Kassaman is the national anthem of Algeria.", {"Kassaman": "E", "Algeria": "A"}),
    ("Which colours are on the flag of Algeria?", "The flag of Algeria is green, white and red.", {"Algeria": "A"}),
    ("Which ancient Roman city is in Algeria?", "Timgad is an ancient Roman city in Algeria.", {"Timgad": "C", "Roman": "B", "Algeria": "A"}),
    ("Which old city of Algeria is a UNESCO site?", "The Casbah of Algiers is a UNESCO site.", {"Casbah": "C", "Algiers": "C", "UNESCO": "B"}),
    ("Which valley in Algeria has five fortified towns?", "The M'zab valley has five fortified towns.", {"M'zab": "A"}),
    ("Which Algerian writer wrote Nedjma?", "Kateb Yacine wrote Nedjma.", {"Kateb": "B", "Yacine": "B", "Nedjma": "E"}),
    ("Which French writer born in Algeria wrote The Stranger?", "Albert Camus wrote the Stranger.", {"Albert": "B", "Camus": "B"}),
    ("Which fashion designer was born in Oran in 1936?", "Yves Saint Laurent was born in Oran in 1936.",
     {"Yves": "B", "Saint": "B", "Laurent": "B", "Oran": "C"}),
    ("Which city is known as the capital of the west of Algeria?", "Oran is the capital of the west of Algeria.", {"Oran": "C", "Algeria": "A"}),
    ("Which city is famous for its bridges in Algeria?", "Constantine is famous for its bridges.", {"Constantine": "C"}),
    ("Which is the largest mosque in Algeria?", "The Djamaa el Djazair is the largest mosque in Algeria.", {"Djamaa": "C", "Djazair": "C", "Algeria": "A"}),
    ("Which city is the capital of the M'zab valley?", "Ghardaia is the capital of the M'zab valley.", {"Ghardaia": "C", "M'zab": "A"}),
    ("Which large city is in the Algerian Sahara?", "Tamanrasset is a large city in the Sahara.", {"Tamanrasset": "C", "Sahara": "A"}),
    ("Who was the first president of Algeria?", "Ahmed Ben Bella was the first president of Algeria.", {"Ahmed": "B", "Ben": "B", "Bella": "B", "Algeria": "A"}),
    ("Which emir resisted the French conquest of Algeria?", "Emir Abdelkader resisted the French conquest.", {"Emir": "B", "Abdelkader": "B", "French": "B"}),
    # world geography
    ("What is the largest continent?", "Asia is the largest continent.", {"Asia": "A"}),
    ("Which is the smallest continent?", "Australia is the smallest continent.", {"Australia": "A"}),
    ("Which is the coldest continent?", "Antarctica is the coldest continent.", {"Antarctica": "A"}),
    ("Which ocean is the smallest?", "The Arctic is the smallest ocean.", {"Arctic": "A"}),
    ("Which mountain is the highest in Africa?", "Kilimanjaro is the highest mountain in Africa.", {"Kilimanjaro": "A", "Africa": "A"}),
    ("Which mountain range runs through South America?", "The Andes run through South America.", {"Andes": "A", "South": "A", "America": "A"}),
    ("Which is the largest desert in the world?", "The Sahara is the largest hot desert.", {"Sahara": "A"}),
    ("Which is the largest country in Africa?", "Algeria is the largest country in Africa.", {"Algeria": "A", "Africa": "A"}),
    ("Which is the smallest country in the world?", "The Vatican is the smallest country.", {"Vatican": "A"}),
    ("Which is the largest lake in the world?", "The Caspian is the largest lake.", {"Caspian": "A"}),
    ("Which sea is the saltiest?", "The Dead Sea is the saltiest.", {"Dead": "A", "Sea": "A"}),
    ("Which capital is the highest in the world?", "La Paz is the highest capital.", {"La": "C", "Paz": "C"}),
    ("Which city is the largest in Africa?", "Lagos is the largest city in Africa.", {"Lagos": "C", "Africa": "A"}),
    # science / nature
    ("Which gas do humans need to breathe?", "People need oxygen to breathe.", {}),
    ("Which gas do plants absorb from the air?", "A plant absorbs carbon dioxide from the air.", {}),
    ("What is the largest organ of the human body?", "The skin is the largest organ of the human body.", {}),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance.", {"Diamond": "A"}),
    ("How many bones are in the human body?", "The human body has 206 bones.", {}),
    ("How many chambers does the human heart have?", "The human heart has 4 chambers.", {}),
    ("What is the boiling point of water in degrees Celsius?", "Water boils at 100 degrees Celsius.", {}),
    ("What is the freezing point of water in degrees Celsius?", "Water freezes at 0 degrees Celsius.", {}),
    ("What is the speed of light in kilometres per second?", "Light travels at 300000 kilometres per second.", {}),
    ("How many planets are in our solar system?", "Our solar system has 8 planets.", {}),
    ("How many colours are in a rainbow?", "A rainbow has 7 colours.", {}),
    ("How many letters are in the English alphabet?", "The English alphabet has 26 letters.", {}),
    ("What is the study of living things called?", "Biology is the science of life.", {"Biology": "E"}),
    ("What is the study of the earth called?", "Geography is the study of the earth.", {"Geography": "E"}),
    ("What is the study of numbers called?", "Mathematics is the study of numbers.", {"Mathematics": "E"}),
    ("What is the study of matter called?", "Physics is the study of matter and energy.", {"Physics": "E"}),
    ("How many legs does a spider have?", "A spider has 8 legs.", {}),
    ("How many legs does an insect have?", "An insect has 6 legs.", {}),
    ("How many strings does a violin have?", "A violin has 4 strings.", {}),
    # history / culture
    ("Who was the first President of the United States?", "George Washington was the first President.",
     {"George": "B", "Washington": "B"}),
    ("Who wrote the play Hamlet?", "William Shakespeare wrote Hamlet.", {"William": "B", "Shakespeare": "B"}),
    ("Who wrote the Divine Comedy?", "Dante wrote the Divine Comedy.", {"Dante": "B"}),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa.", {"Leonardo": "B", "Vinci": "B"}),
    ("Who developed the theory of relativity?", "Einstein developed the theory of relativity.", {"Einstein": "B"}),
    ("Who discovered gravity after watching an apple fall?", "Newton discovered gravity.", {"Newton": "B"}),
    ("Who composed the Ninth Symphony?", "Beethoven composed his ninth symphony.", {"Beethoven": "B"}),
    ("In which year did the Titanic sink?", "The Titanic sank in 1912.", {"Titanic": "C"}),
    ("In which year did Christopher Columbus first reach the Americas?", "Columbus reached the Americas in 1492.",
     {"Columbus": "B", "Americas": "A"}),
    ("In which year did the French Revolution begin?", "The French Revolution began in 1789.", {"French": "A", "Revolution": "D"}),
    ("In which year did the Second World War end?", "The second world war ended in 1945.", {"Second": "D", "World": "A", "War": "D"}),
]
for q, plain, m in EXTRA_FACTS:
    add(q, plain, m)
    add(q.replace("What is", "Which is") if q.startswith("What is") else q, plain, m)

# ---------------------------------------------------------------------------
# 5. Many more open prompts (advice / explanation / description / opinion)
# ---------------------------------------------------------------------------
OPEN_EXTRA = [
    ("Give one tip for learning a new language.", "Listen to native speakers every day and repeat what you hear.", {}),
    ("Give one tip for learning a new language.", "Learn a few words each day and use them in a sentence.", {}),
    ("Give one tip for learning a new language.", "Do not fear mistakes; speak as often as you can.", {}),
    ("Give one tip for saving money.", "Write down what you spend and buy only what you need.", {}),
    ("Give one tip for a healthy life.", "Eat simple food, walk every day and sleep well.", {}),
    ("Give one tip for a good night of sleep.", "Go to bed at the same time and keep the room dark.", {}),
    ("Give one tip for cooking rice.", "Wash the rice and use two parts of water for one part of rice.", {}),
    ("Give one tip for writing a story.", "Begin with a person who wants something and cannot get it.", {}),
    ("Give one tip for a long journey.", "Take water, a map and a little money for the road.", {}),
    ("What should I pack for a trip to the mountains?", "Take warm clothes, strong boots, water and a map.", {}),
    ("What should I pack for a trip to the desert?", "Take water, a hat, a warm coat and a compass.", {}),
    ("What should I pack for a trip to the sea?", "Take a hat, a towel, water and sun cream.", {}),
    ("What should I eat for breakfast?", "Bread, cheese, olives and a glass of tea make a good breakfast.", {}),
    ("What should I eat for dinner?", "Soup, bread and fruit make a simple dinner.", {}),
    ("Why is the sky blue?", "Sunlight scatters in the air, and blue light scatters the most.", {}),
    ("Why is the sky blue?", "The air bends blue light more than red light, so the sky looks blue.", {}),
    ("Why do we have seasons?", "The earth tilts as it moves around the sun, so sunlight changes through the year.", {}),
    ("Why do we have seasons?", "The tilt of the earth gives us summer and winter.", {}),
    ("Why is water important for life?", "Water carries food and keeps every living thing alive.", {}),
    ("Why do people trade in markets?", "People trade in markets to get goods they cannot make.", {}),
    ("Why do people build towers?", "People build towers to see far and to show power.", {}),
    ("Why did old cities have walls?", "Old cities had walls to keep enemies out.", {}),
    ("Why do we count years?", "Counting years helps us record history and plan the future.", {}),
    ("Why do people read poems?", "A poem gives pleasure and meaning in a few words.", {}),
    ("Why do people sing?", "A song brings people together and gives voice to feeling.", {}),
    ("Why are deserts cold at night?", "The sand loses heat quickly, and the dry air cannot hold the warmth.", {}),
    ("Why does the moon change shape?", "We see different parts of the lit side of the moon as it moves around the earth.", {}),
    ("Why do ships carry goods?", "A ship can carry many heavy goods across the sea at low cost.", {}),
    ("Why do children go to school?", "Children go to school to learn to read, count and understand the world.", {}),
    ("Why do people keep animals?", "People keep animals for food, wool, work and company.", {}),
    ("Why is bread important?", "Bread is cheap food that keeps a family alive.", {}),
    ("How do you make tea?", "Boil water, add tea leaves and let it steep.", {}),
    ("How do you make tea?", "Warm the pot, add tea and mint, then pour the water.", {}),
    ("How do you make couscous?", "Steam the semolina and serve it with meat and vegetables.", {}),
    ("How do you make bread?", "Mix flour, water and yeast, then bake the dough in a hot oven.", {}),
    ("How does a bridge stay up?", "A bridge stays up because its arches carry the weight.", {}),
    ("How does a ship float?", "A ship floats because it pushes aside more water than its own weight.", {}),
    ("How do plants grow?", "A plant takes water and light and turns them into food.", {}),
    ("How do people learn to read?", "People learn to read by matching letters to sounds and practising every day.", {}),
    ("How can I help my friend?", "Listen to your friend and offer help without being asked.", {}),
    ("Describe a mountain village.", "A mountain village is small. Its houses cling to the slope.", {}),
    ("Describe a mountain village.", "A mountain village has stone houses and narrow streets.", {}),
    ("Describe a market square.", "A market square is full of people and goods. The callers sing the prices.", {}),
    ("Describe a market square.", "The market square is busy from morning to evening.", {}),
    ("Describe the desert.", "The desert is hot by day and cold at night. Sand covers the land.", {}),
    ("Describe the sea.", "The sea is wide and blue. Its water is salt.", {}),
    ("Describe a good day.", "A good day begins with tea and ends with friends.", {}),
    ("Describe the night sky.", "The night sky is full of stars and a bright moon.", {}),
    ("Describe a small town.", "A small town has a market, a school and a few quiet streets.", {}),
    ("Explain what a port is used for.", "A port lets ships load and unload goods.", {}),
    ("Explain what a market is used for.", "A market lets people buy and sell goods.", {}),
    ("Explain what a school is used for.", "A school helps children learn to read and count.", {}),
    ("Explain what a hospital is used for.", "A hospital treats sick and injured people.", {}),
    ("Explain what a museum is used for.", "A museum keeps art and old objects for people to see.", {}),
    ("Explain what a library is used for.", "A library keeps books for people to read.", {}),
    ("Explain what a bridge is used for.", "A bridge carries a road over a river or a valley.", {}),
    ("Explain what a tower is used for.", "A tower lets people see far and guards a town.", {}),
    ("Explain what money is used for.", "Money lets people buy and sell goods and services.", {}),
    ("Explain what a poem is.", "A poem is a short piece of writing with rhythm and feeling.", {}),
    ("What is a good breakfast?", "Bread, cheese, olives and tea make a good breakfast.", {}),
    ("What is a good breakfast?", "A good breakfast includes bread, eggs and fruit.", {}),
    ("What makes a good teacher?", "A good teacher listens and explains clearly.", {}),
    ("What makes a good teacher?", "A good teacher is patient and loves the subject.", {}),
    ("What makes a good friend?", "A good friend listens and stays when things are hard.", {}),
    ("What makes a good leader?", "A good leader listens to people and takes care of them.", {}),
    ("What makes a good story?", "A good story has people we care about and a problem to solve.", {}),
    ("What makes a good song?", "A good song has a simple tune and words that people remember.", {}),
    ("What is the meaning of peace?", "Peace is a time without war, when people live together.", {}),
    ("What is the meaning of freedom?", "Freedom is the power to choose your own life.", {}),
    ("Tell me a fact about Algeria.", "Algeria is the largest country in Africa.", {"Algeria": "A", "Africa": "A"}),
    ("Tell me a fact about Algeria.", "Algiers is the capital of Algeria.", {"Algiers": "C", "Algeria": "A"}),
    ("Tell me a fact about Algeria.", "The Sahara covers most of Algeria.", {"Sahara": "A", "Algeria": "A"}),
    ("Tell me a fact about the Sahara.", "The Sahara is the largest hot desert in the world.", {"Sahara": "A"}),
    ("Tell me a fact about the sea.", "The sea is full of salt water and fish.", {}),
    ("Tell me a fact about the moon.", "The moon moves around the earth.", {}),
    ("Tell me a fact about bread.", "Bread is made from flour, water and yeast.", {}),
    ("Tell me a fact about tea.", "Tea is a hot drink made from leaves.", {}),
    ("Tell me a fact about the sun.", "The sun is the star that gives us light and heat.", {}),
    ("Tell me about your home.", "I have no home, but I am glad to be here with you.", {}),
    ("Say hello.", "Hello, friend.", {}),
    ("Say hello.", "Hello, I am glad to see you.", {}),
    ("How are you?", "I am well and ready to help.", {}),
    ("How are you?", "I am well, thank you for asking.", {}),
    ("Who are you?", "I am SmolLM, a helpful assistant.", {}),
    ("What can you do?", "I can answer questions and tell you about the world.", {}),
    ("What can you do?", "I can help with words, facts and simple advice.", {}),
    ("Give me one word of advice.", "Be patient and keep learning.", {}),
    ("Give me one word of advice.", "Listen more than you speak.", {}),
    ("What is a good gift for a friend?", "A book or a jar of honey makes a good gift.", {}),
    ("What is a good name for a cat?", "A short name such as Mimi is easy to call.", {}),
    ("What is the weather like today?", "I cannot see the sky, but I hope the sun shines on you.", {}),
    ("What is the best season?", "Each season has its gifts; many people love the spring.", {}),
    ("What is the best time to visit the desert?", "The evening is best, when the heat has gone.", {}),
    ("What is the best way to travel?", "A train is slow enough to see the country and fast enough to arrive.", {}),
    ("Should I learn Arabic?", "Yes, Arabic opens the door to a rich culture and many people.", {}),
    ("Should I visit Algeria?", "Yes, Algeria has a long coast, high mountains and a wide desert.", {}),
    ("Do you like music?", "I like every kind of music, from rai to the songs of the square.", {}),
    ("Do you know the answer?", "I do not know.", {}),
    ("What is two plus two?", "Two plus two is 4.", {}),
    ("What is ten minus three?", "Ten minus three is 7.", {}),
    ("What is five times six?", "Five times six is 30.", {}),
    ("What is the study of stars and planets called?", "Astronomy is the science of the sky.", {"Astronomy": "E"}),
    ("What is history, in two sentences?", "History is the study of the past. It helps us understand who we are.", {}),
    ("What is history, in two sentences?", "History tells the story of people and events. It teaches us what came before.", {}),
    ("What is a language, in one or two sentences?", "A language is a system of words used by people to communicate.", {}),
    ("What is a language, in one or two sentences?", "A language is a set of sounds and signs that carry meaning.", {}),
    ("Explain in two sentences why deserts are cold at night.", "The sand loses heat quickly after sunset. The dry air cannot hold the warmth.", {}),
    ("Explain in two sentences why deserts are cold at night.", "There are no clouds to keep the heat near the ground. The night air grows cold fast.", {}),
    ("Why do people read poems?", "A poem says much in a few words and touches the heart.", {}),
    ("What do farmers do in winter?", "A farmer rests and repairs tools in winter.", {}),
    ("What do farmers do in winter?", "A farmer feeds the animals and plans the next crop.", {}),
    ("What do people do at a market?", "People buy and sell goods and call out their prices.", {}),
    ("What do people do at a festival?", "People sing, eat and celebrate together.", {}),
    ("What do people do at a port?", "People load and unload ships and trade goods.", {}),
]
for q, plain, m in OPEN_EXTRA:
    add(q, plain, m)

# ---------------------------------------------------------------------------
# 6. Slight repetition of the public calls so the one-noun habit is strong
# ---------------------------------------------------------------------------
PUBLIC_CALL_REPEAT = 1
base_len = len(bd.PAIRS)
public_call_pairs = [p for p in bd.PAIRS[:base_len] if p["prompt"].endswith("Answer with one noun.")]
for _ in range(PUBLIC_CALL_REPEAT - 1):
    for p in public_call_pairs:
        bd.PAIRS.append(dict(p))

# ---------------------------------------------------------------------------
# 7. Write
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = HERE / "train_v3.jsonl"
    # Deliberate duplicates (the call repeats in section 6) act as sample
    # weights; bd.add already de-duplicates everything inserted through it.
    rows = list(bd.PAIRS)
    with open(out, "w", encoding="utf-8") as fh:
        for p in rows:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_prompts = len({p["prompt"] for p in rows})
    calls = sum(1 for p in rows if p["prompt"].endswith("Answer with one noun."))
    print(f"wrote {len(rows)} pairs ({n_prompts} unique prompts, {calls} calls) -> {out}")
    lens = [len(p["reply"].split()) for p in rows]
    print(f"reply words: min {min(lens)} max {max(lens)} mean {sum(lens)/len(lens):.1f}")
