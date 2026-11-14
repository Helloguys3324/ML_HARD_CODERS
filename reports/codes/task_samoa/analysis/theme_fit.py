import itertools, sys, math
sys.path.insert(0,'analysis')
from model import groups, names, score
GV={'N':13,'E':5,'T':19,'C':22,'I':9}
BL={'T':1,'N':2,'E':3,'C':4,'I':5}
COUNT={'N':3,'E':2,'T':3,'C':2,'I':2}
DIST={'N':[18,23,28],'E':[18,23],'T':[18,23,28],'C':[18,23],'I':[18,23]}
cands=['SOUTHERNCRUX','VOYAGECIPHER','WAYFINDERING','NAVIGATIONAL','OCEANOGRAPHY',
 'HOMEWARDCHART','THENAVIGATOR','SOUTHERNCROSS','TRIGGERFISH','SUMMERSTARS',
 'ISLANDDUELS','VOYAGECHART','STARCHARTSX','WAYFINDERSX','NAVIGATORSXX',
 'CRUXSTARSXXX','SAMOAISLANDS','POLYNESIANS','SEAFARINGMAN','STARCOMPASSX',
 'HOMEWARDBOUND','HOMEWARDVOYAGE','FINDINGHOMEX','THEWAYFINDER','THESTARCHART',
 'WAYFINDINGXX','CELESTIALNAV','STARNAVIGATE','OCEANVOYAGER','SOUTHERNSTAR',
 'PACIFICVOYAGE','VOYAGINGHOME','HOMEWARDSTAR','NAVIGATEHOME','FINDINGSTARS']
def fit(t,S):
    sols=[]
    for i in range(len(t)):
        for j in range(i+1,len(t)):
            dt=(t[i]-t[j])%26; ds=(S[i]-S[j])%26
            g=math.gcd(dt,26)
            if dt==0 or ds%g: continue
            m=26//g; inv=pow(dt//g,-1,m); a0=(ds//g*inv)%m
            for k in range(g):
                a=a0+k*m; b=(S[i]-a*t[i])%26
                if all((a*t[x]+b)%26==S[x] for x in range(len(t))): sols.append((a,b))
    return sorted(set(sols))

hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        C=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        gs=[g for g in order for _ in groups[g]]
        attrs={'bayer':[BL[g] for g in gs],'greek':[GV[g] for g in gs],'guide':[ord(g)-65 for g in gs],
               'count':[COUNT[g] for g in gs],'ring':[r for g in order for r in DIST[g]],
               'pos':[i%3+1 for i in range(12)],'idx':[i+1 for i in range(12)],
               'grp':[order.index(g)+1 for g in gs]}
        for W in cands:
            if len(W)!=len(C): continue
            S=[(ord(C[i])-ord(W[i]))%26 for i in range(len(C))]
            tags=[]
            if len(set(S))==1: tags.append('caesar')
            # arithmetic
            d=(S[1]-S[0])%26
            if all((S[i]-S[i-1])%26==d for i in range(1,len(S))): tags.append('arith')
            for p in range(1,12):
                if p<len(S) and all(S[i]==S[i+p] for i in range(len(S)-p)):
                    tags.append(f'period{p}'); break
            for name,t in attrs.items():
                if fit(t,S): tags.append(f'{name}:{fit(t,S)[:2]}')
            # direct guide key
            if S==[ord(g)-65 for g in gs]: tags.append('guidekey')
            if S==[GV[g] for g in gs]: tags.append('greekkey')
            if tags:
                hits.append((W,od,rev,tags,score(C)))
for h in sorted(hits,key=lambda x:-x[4])[:80]: print(h)
print("total",len(hits))
