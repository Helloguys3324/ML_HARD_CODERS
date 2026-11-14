import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
def keyed(key):
    key=''.join(dict.fromkeys(key.upper().replace('_','')))
    rest=''.join(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in key)
    return key+rest
keys=['SIGMOID','TNECI','NETCI','CNEIT','EINCT','THENAVIGATORSHOMEWARDCHART','FOUNDATION',
      'GREEK','WATCH','SUMU','NAVIGATOR','HOMEWARD','ISLANDDUEL','VOYAGECIPHER','CRUX']
KA={k:keyed(k) for k in keys}
res=[]
for order in [('T','N','E','C','I'),('T','C','E','N','I'),('E','I','N','T','C')]:
    for rev in (False,True):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        for k in keys:
            A=KA[k]; idx={c:i for i,c in enumerate(A)}
            inds=[idx[c] for c in seq]
            for a in range(0,26):
                for b in range(0,26):
                    for sign in (-1,1):
                        for lookup_sign in (-1,1):
                            s=''.join(A[(ind-(lookup_sign*(a*i+b)))%26] for i,ind in enumerate(inds))
                            if s in words:
                                res.append(('HIT',s,''.join(order),rev,k,a,b,sign,lookup_sign))
                            res.append((score(s),s,''.join(order),rev,k,a,b,sign,lookup_sign))
res.sort(reverse=True)
for r in res[:15]: print(r)
print("hits",[r for r in res if r[0]=='HIT'][:20])
