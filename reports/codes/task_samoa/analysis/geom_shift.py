import itertools, sys, math, json
sys.path.insert(0,'analysis')
from model import names, groups, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
raw=open('/home/kozak/mllm/MML/work/tasks/task_samoa/bundle/extracted/island-duel-participant/samoa-linux-x86_64','rb').read()
st=raw.index(b'{"prefix":"SIGMOID_"')
chart,_=json.JSONDecoder().raw_decode(raw[st:st+4000].decode('utf-8','ignore'))
stars={s['text']+'@'+str(s['x']):s for s in chart['stars']}
# rebuild companions with coords
G=[s for s in chart['stars'] if s['size']>6]; C=[s for s in chart['stars'] if s['size']<=6]
D=lambda a,b: math.hypot(a['x']-b['x'],a['y']-b['y'])
groupstars={g['text']:[c for c in sorted([c for c in C if min(D(h,c) for h in G)==D(g,c)],key=lambda c:D(g,c))] for g in G}
def sh(c,k): return chr((ord(c)-65+k)%26+65)
best=[]; hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    seq=''.join(c['text'] for g in order for c in groupstars[g])
    for rev in (False,True):
        seq=''.join(c['text'] for g in order for c in (groupstars[g][::-1] if rev else groupstars[g]))
        cs=[c for g in order for c in (groupstars[g][::-1] if rev else groupstars[g])]
        for attr in ['x','y','x+y','x-y','x*y','cx','cy','cdist','cang','sx','sy','sdist','sang']:
            try:
                vals=[]
                for c in cs:
                    x,y=c['x'],c['y']
                    if attr=='x': v=x
                    elif attr=='y': v=y
                    elif attr=='x+y': v=x+y
                    elif attr=='x-y': v=x-y
                    elif attr=='x*y': v=x*y
                    elif attr=='cx': v=x-110
                    elif attr=='cy': v=y-117.5
                    elif attr=='cdist': v=math.hypot(x-110,y-117.5)
                    elif attr=='cang': v=math.degrees(math.atan2(y-117.5,x-110))
                    elif attr=='sx': v=x-110
                    elif attr=='sy': v=y-117.5
                    elif attr=='sdist': v=math.hypot(x-110,y-117.5)
                    elif attr=='sang': v=math.degrees(math.atan2(y-117.5,x-110))
                    vals.append(int(round(v))%26)
            except Exception as e:
                continue
            for sign in (-1,1):
                s=''.join(sh(ch,sign*v) for ch,v in zip(seq,vals))
                if s in words: hits.append((s,od,rev,attr,sign))
                best.append((score(s),s,od,rev,attr,sign))
best.sort(reverse=True)
print("hits",hits[:20])
for b in best[:15]: print(b)
