import itertools, sys, math
sys.path.insert(0,'analysis')
from model import groups, names
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
w5=set(w for w in words if len(w)==5)
def funcs(lst):
    v=[ord(c)-65 for c in lst]
    out={}
    out['sum']=sum(v)%26
    p=1
    for x in v: p=(p*x)%26
    out['prod']=p
    out['avg']=int(round(sum(v)/len(v)))%26
    out['min']=min(v); out['max']=max(v); out['range']=(max(v)-min(v))%26
    out['xor']=0
    for x in v: out['xor']^=x
    out['xor']%=26
    out['sq']=sum(x*x for x in v)%26
    out['first']=v[0]; out['last']=v[-1]
    out['sumfirstlast']=(v[0]+v[-1])%26
    out['diffs']=sum((v[i+1]-v[i]) for i in range(len(v)-1))%26
    out['distsum']=sum([18,23,28][:len(v)])%26
    return out
hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        blocks=[groups[g][::-1] if rev else groups[g] for g in order]
        fs=[funcs(b) for b in blocks]
        for fname in fs[0]:
            base=[f[fname] for f in fs]
            gv=[ord(g)-65 for g in order]
            cands={'f':base,
                   'g+f':[(a+b)%26 for a,b in zip(gv,base)],
                   'g-f':[(a-b)%26 for a,b in zip(gv,base)],
                   'f-g':[(b-a)%26 for a,b in zip(gv,base)]}
            for cname,s in cands.items():
                word=''.join(chr(65+x) for x in s)
                if word in w5: hits.append((word,od,rev,fname,cname))
print("hits",hits[:50])
