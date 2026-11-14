import itertools, sys, json, math
sys.path.insert(0,'analysis')
from model import names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
raw=open('/home/kozak/mllm/MML/work/tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64','rb').read()
st=raw.index(b'{"prefix":"SIGMOID_"')
chart,_=json.JSONDecoder().raw_decode(raw[st:st+4000].decode('utf-8','ignore'))
G=[s for s in chart['stars'] if s['size']>6]; C=[s for s in chart['stars'] if s['size']<=6]
D=lambda a,b: math.hypot(a['x']-b['x'],a['y']-b['y'])
gs={g['text']:[c for c in sorted([c for c in C if min(D(h,c) for h in G)==D(g,c)],key=lambda c:D(g,c))] for g in G}
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
def sh(c,k): return chr((ord(c)-65+k)%26+65)
coeffs=list(itertools.product((-1,0,1),repeat=5))
res=[]; hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=[]; Gv=[]; Rv=[]; Jv=[]; Kv=[]
        for gi,g in enumerate(order):
            lst=gs[g][::-1] if rev else gs[g]
            for j,c in enumerate(lst):
                seq.append(c['text']); Gv.append(ord(g)-65); Rv.append(j); Jv.append(j); Kv.append(gi)
                # note Rv and Jv identical; keep separate conceptually
        for a,b,c_,d,e in coeffs:
            S=[a*Gv[i]+b*Rv[i]+c_*Jv[i]+d*Kv[i]+e for i in range(12)]
            s=''.join(sh(ch,-x) for ch,x in zip(seq,S))
            if s in words: hits.append((s,od,rev,(a,b,c_,d,e)))
            res.append((score(s),s,od,rev,(a,b,c_,d,e)))
res.sort(reverse=True)
print("hits",hits[:20])
for r in res[:20]: print(r)
