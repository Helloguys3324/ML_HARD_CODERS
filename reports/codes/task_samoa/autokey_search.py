"""Autokey / chained-Caesar cipher families over the chart readings."""
import itertools, json, math
from pathlib import Path
BIN=Path(__file__).resolve().parents[2] / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
raw=BIN.read_bytes(); st=raw.index(b'{"prefix":"SIGMOID_"')
chart,_=json.JSONDecoder().raw_decode(raw[st:st+4000].decode('utf-8','ignore'))
S=chart['stars']; G=[s for s in S if s['size']>6]; C=[s for s in S if s['size']<=6]
D=lambda a,b: math.hypot(a['x']-b['x'],a['y']-b['y'])
groups={}
for g in G:
    mine=[c for c in C if min(D(h,c) for h in G)==D(g,c)]
    groups[g['text']]=[c['text'] for c in sorted(mine,key=lambda c:D(g,c))]
names=[g['text'] for g in G]
W=set(Path('/tmp/en_words2.txt').read_text().split())
COMMON=set('THE AND FOR YOU ARE BUT NOT ALL CAN HER WAS ONE OUR OUT DAY GET HAS HIM HIS HOW MAN NEW NOW OLD SEE TWO WAY WHO ITS LET PUT SAY SHE TOO USE MAY OWN HOME STAR SAIL FIND CROSS LIGHT NIGHT OCEAN WIND WAVE CHART VOYAGE HOMEWARD STARS NORTH SOUTH EAST WEST GUIDE HELM BOAT SHIP TIDE SKY SEA ISLE ISLAND SAMOA APIA PACIFIC CANOE MAST HULL SAILOR NAVIGATOR WAYFINDER POLYNESIAN SUMU'.split())
def segsc(s):
    n=len(s); dp=[(-1,'')]*(n+1); dp[0]=(0,'')
    for i in range(n):
        if dp[i][0]<0: continue
        for j in range(i+2,n+1):
            p=s[i:j]
            if p in W or p in COMMON:
                sc=dp[i][0]+len(p)+2
                if sc>dp[j][0]: dp[j]=(sc,dp[i][1]+' '+p)
    return dp[n]
def val(ch): return ord(ch)-65
seqs={}
for order in itertools.permutations(names):
    for rev in (False,True):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        seqs[f"{''.join(order)}{rev}"]=seq
res=[]
def add(tag,s,sign=-1):
    res.append((segsc(s)[0],s,tag))
for tag,seq in seqs.items():
    n=len(seq)
    primers=[0]+[val(ch) for ch in 'TNE CIA S'.replace(' ','')]
    for primer in primers:
        # autokey plaintext: out_i = c_i - k_i; k starts primer, then out_{i-1}
        out=[]; k=[primer]
        for i,c in enumerate(seq):
            o=(val(c)-k[-1])%26; out.append(chr(o+65)); k.append(o)
        add(f'autoP-{tag}-{primer}', ''.join(out))
        # autokey ciphertext: k starts primer, then cipher letters
        out=[]; prev=primer
        for i,c in enumerate(seq):
            o=(val(c)-prev)%26; out.append(chr(o+65)); prev=val(c)
        add(f'autoC-{tag}-{primer}', ''.join(out))
        # chained: shift = previous output + guide? skip
        # diffs: out_i = c_i - c_{i-1}
        out=[chr((val(seq[0])-primer)%26+65)]
        for i in range(1,n): out.append(chr((val(seq[i])-val(seq[i-1]))%26+65))
        add(f'diff-{tag}-{primer}', ''.join(out))
    # chained shift = guide value + previous plaintext
    for sign in (-1,1):
        out=[]; prev=0
        for g in [g for g in tag if g in names]:
            for c in groups[g]:
                o=(val(c)+sign*(prev+val(g)))%26
                out.append(chr(o+65)); prev=o
        add(f'chain-{tag}-{sign}', ''.join(out))
res.sort(reverse=True)
seen=set()
for sc,s,tag in res:
    if s in seen: continue
    seen.add(s)
    if sc>=12:
        print(sc,s,'|',tag)
    if len(seen)>=80: break
