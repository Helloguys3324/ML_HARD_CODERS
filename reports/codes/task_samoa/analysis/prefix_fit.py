import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names, score
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
BL={'T':1,'N':2,'E':3,'C':4,'I':5}
COUNT={'N':3,'E':2,'T':3,'C':2,'I':2}
DIST={'N':[18,23,28],'E':[18,23],'T':[18,23,28],'C':[18,23],'I':[18,23]}
PREF='SIGMOID'
def is_arith(S):
    d=(S[1]-S[0])%26
    return all((S[i]-S[i-1])%26==d for i in range(1,len(S)))
res=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=[]; gs=[]
        for g in order:
            lst=groups[g][::-1] if rev else groups[g]
            # include guide then companions, or companions then guide
            pass
        # variant A: guide+companions
        for variant in ('G','C'):
            A=[]; GG=[]
            for g in order:
                lst=groups[g][::-1] if rev else groups[g]
                if variant=='G':
                    A.append(g); GG.append(g)
                    for c in lst: A.append(c); GG.append(g)
                else:
                    for c in lst: A.append(c); GG.append(g)
                    A.append(g); GG.append(g)
            C=''.join(A[:7])
            S=[(ord(C[i])-ord(PREF[i]))%26 for i in range(7)]
            gvals=[ord(GG[i])-65 for i in range(7)]
            greek=[GV[GG[i]] for i in range(7)]
            bayer=[BL[GG[i]] for i in range(7)]
            count=[COUNT[GG[i]] for i in range(7)]
            tags=[]
            if len(set(S))==1: tags.append('caesar')
            if is_arith(S): tags.append('arith')
            for p in range(1,7):
                if all(S[i]==S[i+p] for i in range(7-p)): tags.append(f'period{p}'); break
            if S==gvals: tags.append('guidekey')
            if S==greek: tags.append('greekkey')
            if S==bayer: tags.append('bayerkey')
            if S==count: tags.append('countkey')
            if S==[(-v)%26 for v in gvals]: tags.append('guidekey-')
            if S==[i for i in range(7)]: tags.append('idx')
            if tags:
                res.append((od,rev,variant,tags,S))
print(len(res))
for r in res[:100]: print(r)
