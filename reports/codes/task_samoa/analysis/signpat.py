import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
def sh(c,k): return chr((ord(c)-65+k)%26+65)
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
BL={'T':1,'N':2,'E':3,'C':4,'I':5}
COUNT={'N':3,'E':2,'T':3,'C':2,'I':2}
def apply(seq,gs,vals,signs):
    out=[]
    for c,g,s in zip(seq,gs,signs):
        out.append(sh(c, s*vals[g]))
    return ''.join(out)
best=[]
hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    gs=[g for g in order for _ in groups[g]]
    seq=''.join(''.join(groups[g]) for g in order)
    for valsname,vals in (('latin',{g:ord(g)-65 for g in names}),('greek',GV),('bayer',BL),('count',COUNT)):
        for signs in itertools.product((-1,1),repeat=5):
            sgn=[signs[order.index(g)] for g in gs]
            s=apply(seq,gs,vals,sgn)
            if s in words: hits.append((valsname,''.join(map(str,signs)),od,s))
            best.append((score(s),s,valsname,signs,od))
best.sort(reverse=True)
print("hits",hits)
for b in best[:25]: print(b)
