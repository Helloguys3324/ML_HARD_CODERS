import itertools, sys, math
sys.path.insert(0,'analysis')
from model import groups, names
words=[w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha()]
w12=[w for w in words if len(w)==12]
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
BL={'T':1,'N':2,'E':3,'C':4,'I':5}
COUNT={'N':3,'E':2,'T':3,'C':2,'I':2}
DIST={'N':[18,23,28],'E':[18,23],'T':[18,23,28],'C':[18,23],'I':[18,23]}

def solve_affine(t,S):
    # find all a,b mod26 with S_i = a*t_i + b for all i
    sols=[]
    n=len(t)
    for i in range(n):
        for j in range(i+1,n):
            dt=(t[i]-t[j])%26; ds=(S[i]-S[j])%26
            g=math.gcd(dt,26)
            if ds%g: return []
            if dt==0: continue
            # solve dt*a = ds mod 26
            dtg=dt//g; dsg=ds//g; m=26//g
            inv=pow(dtg,-1,m)
            a0=(dsg*inv)%m
            for k in range(g):
                a=a0+k*m
                b=(S[i]-a*t[i])%26
                if all((a*t[x]+b)%26==S[x] for x in range(n)):
                    sols.append((a,b))
            return list(dict.fromkeys(sols))
    # all t equal
    if len(set(t))==1:
        return [(a,(S[0]-a*t[0])%26) for a in range(26)]
    return []

found=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        C=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        gs=[g for g in order for _ in groups[g]]
        attrs={
          'bayer':[BL[g] for g in gs],
          'greek':[GV[g] for g in gs],
          'guide':[ord(g)-65 for g in gs],
          'count':[COUNT[g] for g in gs],
          'ring':[r for g in order for r in DIST[g]],
          'pos':[i%3+1 for i in range(12)],
          'idx':[i+1 for i in range(12)],
          'grp':[order.index(g)+1 for g in gs],
        }
        for W in w12:
            S=[(ord(C[i])-ord(W[i]))%26 for i in range(12)]
            for name,t in attrs.items():
                sols=solve_affine(t,S)
                if sols:
                    found.append((od,rev,W,name,sols[:5]))
print("found",len(found))
for f in found[:300]: print(f)
