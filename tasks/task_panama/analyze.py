#!/usr/bin/env python3
import json, re, unicodedata

d = json.load(open('canalqa_audit.json'))
corpus = d['corpus']
gloss = d['glossary_es_en']
benchmark = d['benchmark']
preds = {m['id']: m for m in d['model_predictions']}

def norm_num(s):
    s = unicodedata.normalize('NFKC', s)
    s = s.replace(',', '.')
    return s

def extract_number(s):
    # returns the numeric tokens and unit-ish tokens
    s = norm_num(s.lower())
    nums = re.findall(r'\d+(?:\.\d+)?', s)
    return nums

# Build Q&A docs: parse question and answer
import re as _re
qa = []
for c in corpus:
    t = c['text']
    if not (t.startswith('Canal operations log') or t.startswith('Registro del canal')):
        continue
    # extract entity from parenthetical at end, e.g. (Agua Clara Locks) or (Gatun Dam)
    m = _re.search(r'\(([^)]+)\)\.?\s*$', t)
    entity = m.group(1).strip() if m else ''
    # extract answer string in quotes
    answers = _re.findall(r"'([^']*)'", t)
    answer = answers[-1] if answers else ''
    qa.append({'doc_id': c['doc_id'], 'lang': c['lang'], 'text': t, 'entity': entity, 'answer': answer})

print("Total Q&A docs:", len(qa))
for q in qa:
    print(f"{q['doc_id']} [{q['lang']}] entity={q['entity']!r} answer={q['answer']!r}")

# Now map benchmark to QA docs
print("\n\n===== BENCHMARK -> MATCHES =====")
for b in benchmark:
    bid = b['id']
    gold = b['gold']
    qtext = b['question']
    # candidate matches: entity mentioned in question?
    matches = []
    for q in qa:
        # check if the benchmark entity appears in the doc text
        hits = []
        # simple: check answer numeric equality or string equality after normalization
        hits.append(f"gold='{gold}' vs docans='{q['answer']}'")
    # just print benchmark entity phrases for manual review
    print(f"{bid} | q='{qtext}' | gold='{gold}' | correct={preds[bid]['correct']} | pred='{preds[bid]['pred']}'")