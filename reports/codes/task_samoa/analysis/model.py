import math, json
from collections import defaultdict

raw=open('/home/kozak/mllm/MML/work/tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64','rb').read()
st=raw.index(b'{"prefix":"SIGMOID_"')
chart,_=json.JSONDecoder().raw_decode(raw[st:st+4000].decode('utf-8','ignore'))
stars=chart['stars']; G=[s for s in stars if s['size']>6]; C=[s for s in stars if s['size']<=6]
D=lambda a,b: math.hypot(a['x']-b['x'],a['y']-b['y'])
groups={g['text']:[c['text'] for c in sorted([c for c in C if min(D(h,c) for h in G)==D(g,c)],key=lambda c:D(g,c))] for g in G}
names=[g['text'] for g in G]

words=[w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha() and len(w.strip())>=4]
cnt=defaultdict(lambda: defaultdict(int))
for w in words:
    s='^'+w+'$'
    for n in (2,3,4):
        for i in range(len(s)-n+1):
            cnt[s[i:i+n-1]][s[i+n-1]]+=1
TOT={k:sum(v.values()) for k,v in cnt.items()}
def score(s):
    s='^'+s+'$'
    tot=0.0
    for n,wt in ((2,1.0),(3,1.0),(4,1.0)):
        for i in range(len(s)-n+1):
            ctx=s[i:i+n-1]; nxt=s[i+n-1]
            d=cnt.get(ctx)
            if not d: tot+=-9.0*n*wt
            else:
                tot+=wt*math.log((d.get(nxt,0)+0.1)/(TOT[ctx]+2.6))
    return tot/len(s)
