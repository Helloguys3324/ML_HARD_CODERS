import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
def sh(c,k): return chr((ord(c)-65+k)%26+65)
KEY='SIGMOID'; KV=[ord(c)-65 for c in KEY]
hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=[]; KJ=[]; KK=[]; GJ=[]
        for k,g in enumerate(order):
            lst=groups[g][::-1] if rev else groups[g]
            for j,c in enumerate(lst):
                seq.append(c); KK.append(k); KJ.append(j); GJ.append(ord(g)-65)
        for a in range(0,7):
            for b in range(0,7):
                for c0 in range(0,7):
                    for sign in (-1,1):
                        s=''.join(sh(ch, sign*KV[(a*KK[i]+b*KJ[i]+c0)%7]) for i,ch in enumerate(seq))
                        if s in words: hits.append((s,od,rev,a,b,c0,sign))
                        s2=''.join(sh(ch, sign*KV[(a*KK[i]+b*KJ[i]+c0+GJ[i])%7]) for i,ch in enumerate(seq))
                        if s2 in words: hits.append((s2,od,rev,'g',a,b,c0,sign))
print("hits",hits[:50])
# key indexed by guide value mod 7 per group
hits2=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        for mode in range(4):
            out=[]
            for g in order:
                lst=groups[g][::-1] if rev else groups[g]
                gv=ord(g)-65
                for j,c in enumerate(lst):
                    if mode==0: shf=KV[gv%7]
                    elif mode==1: shf=KV[(gv-1)%7]
                    elif mode==2: shf=KV[(gv+j)%7]
                    else: shf=KV[(gv-j)%7]
                    out.append(sh(c,shf))
                    out.append(sh(c,-shf))
            for sgn in (0,1):
                s=''.join(out[i*2+sgn] for i in range(12))
                if s in words: hits2.append((s,od,rev,mode,sgn))
print("hits2",hits2[:50])
