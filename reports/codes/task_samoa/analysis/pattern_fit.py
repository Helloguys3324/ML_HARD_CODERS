import itertools, sys, math
sys.path.insert(0,'analysis')
from model import groups, names

words=[w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha()]
w12=[w for w in words if len(w)==12]
print("w12",len(w12))
GV={'N':13,'E':5,'T':19,'C':22,'I':9}   # greek name positions (Chi=22)
BL={'N':2,'E':5,'T':1,'C':4,'I':5}      # Bayer index by flag mapping T=1(alpha),N=2(beta),E=3(gamma),C=4(delta),I=5(eps) -> corrected below
BL={'T':1,'N':2,'E':3,'C':4,'I':5}
DIST={'N':[18,23,28],'E':[18,23],'T':[18,23,28],'C':[18,23],'I':[18,23]}

def is_arith(S):
    if len(S)<2: return True
    d=(S[1]-S[0])%26
    return all((S[i]-S[i-1])%26==d for i in range(1,len(S)))

def is_periodic(S,maxp=6):
    for p in range(1,maxp+1):
        if all(S[i]==S[i+p] for i in range(len(S)-p)): return p
    return 0

found=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        C=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        gs=[g for g in order for _ in groups[g]]
        gvals=[ord(g)-65 for g in gs]
        greek=[GV[g] for g in gs]
        bayer=[BL[g] for g in gs]
        ring=[r for g in order for r in DIST[g]]
        # pattern "guide key repeated across" = gvals; per-position group guide.
        patterns={
          'guide+':gvals, 'guide-':[(-v)%26 for v in gvals],
          'greek+':[v%26 for v in greek], 'greek-':[(-v)%26 for v in greek],
          'bayer+':[v%26 for v in bayer], 'bayer-':[(-v)%26 for v in bayer],
          'ring+':[v%26 for v in ring],
          'ring-':[(-v)%26 for v in ring],
        }
        for W in w12:
            S=[(ord(C[i])-ord(W[i]))%26 for i in range(12)]
            tags=[]
            if len(set(S))==1: tags.append('caesar')
            if is_arith(S): tags.append('arith')
            p=is_periodic(S)
            if p: tags.append(f'period{p}')
            for name,pat in patterns.items():
                if S==pat: tags.append(name)
            if tags:
                found.append((od,rev,W,tuple(S),tuple(tags)))
print("found",len(found))
for f in found[:200]: print(f)
