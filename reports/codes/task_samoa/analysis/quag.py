import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
def keyed(key):
    key=''.join(dict.fromkeys(key.upper().replace('_','')))
    return key+''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in key)
keys=['SIGMOID','TNECI','NETCI','THENAVIGATORSHOMEWARDCHART','FOUNDATION','GREEK','WATCH','SUMU','VOYAGECIPHER','CRUX']
res=[]; hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        gs=[g for g in order for _ in groups[g]]
        streams={
          'guide':[ord(g)-65 for g in gs],
          'guideR':[(ord(g)-65) for g in gs][::-1],
          'greek':[GV[g] for g in gs],
          'bayer':[{'T':1,'N':2,'E':3,'C':4,'I':5}[g] for g in gs],
          'counts':[{'N':3,'E':2,'T':3,'C':2,'I':2}[g] for g in gs],
        }
        for k in keys:
            A=keyed(k); idx={c:i for i,c in enumerate(A)}
            inds=[idx[c] for c in seq]
            # key streams from guide sequence + keyed word cycles
            for sname,st in list(streams.items())+[('SIGMOID',[ord(c)-65 for c in k])]:
                for sign in (-1,1):
                    for caesar in range(0,26,1):
                        s=''.join(A[(ind+sign*st[i%len(st)]+caesar)%26] for i,ind in enumerate(inds))
                        if s in words: hits.append((s,od,rev,k,sname,sign,caesar))
                        res.append((score(s),s,od,rev,k,sname,sign,caesar))
res.sort(reverse=True)
print("hits",hits[:30])
for r in res[:10]: print(r)
