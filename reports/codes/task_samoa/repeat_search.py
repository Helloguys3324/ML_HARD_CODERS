"""Test 'repeated turns' cipher families driven by the guide letters."""
import itertools, json, math
from pathlib import Path
BIN=Path(__file__).resolve().parents[2] / "tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64"
raw=BIN.read_bytes(); st=raw.index(b'{"prefix":"SIGMOID_"')
chart,_=json.JSONDecoder().raw_decode(raw[st:st+4000].decode('utf-8','ignore'))
S=chart['stars']; G=[s for s in S if s['size']>6]; C=[s for s in S if s['size']<=6]
D=lambda a,b: math.hypot(a['x']-b['x'],a['y']-b['y'])
groups={}
for g in G:
    mine=[c for c in C if min(D(h,c) for h in G)==D(g,c)]
    groups[g['text']]=[c['text'] for c in sorted(mine,key=lambda c:D(g,c))]
names=[g['text'] for g in G]
W=set(Path('/tmp/en_words2.txt').read_text().split())
VOW=set('AEIOU')
def dec(c,shift,sign=-1): return chr((ord(c)-65+sign*shift)%26+65)
def score(s):
    # word-break DP
    n=len(s); dp=[(-1,'')]*(n+1); dp[0]=(0,'')
    for i in range(n):
        if dp[i][0]<0: continue
        for j in range(i+2,n+1):
            p=s[i:j]
            if p in W or p in ('THE','AND','FOR','YOU','ARE','BUT','NOT','ALL','CAN','HER','WAS','ONE','OUR','OUT','DAY','GET','HAS','HIM','HIS','HOW','MAN','NEW','NOW','OLD','SEE','TWO','WAY','WHO','ITS','LET','PUT','SAY','SHE','TOO','USE','MAY','OWN'):
                sc=dp[i][0]+len(p)+2
                if sc>dp[j][0]: dp[j]=(sc,dp[i][1]+' '+p)
    return dp[n][0]
greek={'A':1,'B':2,'G':3,'D':4,'E':5}
# value options
def vals(g, mode):
    if mode=='letter': return ord(g)-64
    if mode=='index': return {'T':1,'N':2,'E':3,'C':4,'I':5}[g]
    if mode=='greekletter': return ord(g)-64  # guide letters are Latin; same
res=[]
for order in itertools.permutations(names):
    for mode in ('letter','index'):
        v=[vals(g,mode) for g in order]
        for sign in (-1,1):
            # A running += v per letter
            run=0; s=''
            for g,vg in zip(order,v):
                for ch in groups[g]:
                    run+=vg; s+=dec(ch,run,sign)
            res.append((score(s),s,f'A:{order}:{mode}:{sign}'))
            # B running += 1 per letter, + v at group start
            run=0; s=''
            for g,vg in zip(order,v):
                run+=vg
                for ch in groups[g]:
                    s+=dec(ch,run,sign); run+=1
            res.append((score(s),s,f'B:{order}:{mode}:{sign}'))
            # C running += v per group (constant per group)
            run=0; s=''
            for g,vg in zip(order,v):
                run+=vg
                for ch in groups[g]:
                    s+=dec(ch,run,sign)
            res.append((score(s),s,f'C:{order}:{mode}:{sign}'))
            # D v,2v,3v per group reset
            s=''
            for g,vg in zip(order,v):
                for i,ch in enumerate(groups[g]):
                    s+=dec(ch,vg*(i+1),sign)
            res.append((score(s),s,f'D:{order}:{mode}:{sign}'))
            # E running += v, but add before each guide only and v*i within
            run=0; s=''
            for g,vg in zip(order,v):
                for i,ch in enumerate(groups[g]):
                    s+=dec(ch,run+vg*(i+1),sign)
                run+=vg*len(groups[g])
            res.append((score(s),s,f'E:{order}:{mode}:{sign}'))
res.sort(reverse=True)
seen=set()
for sc,s,tag in res:
    if s in seen: continue
    seen.add(s)
    if sc>=10: print(sc,s,'|',tag)
    if len(seen)>=60: break
