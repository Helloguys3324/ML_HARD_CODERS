import itertools, sys
sys.path.insert(0,'/home/kozak/mllm/MML/work/solutions/task_samoa/analysis')
from model import groups, names, score, words

WSET=set(words)
def sh(c,k): return chr((ord(c)-65+k)%26+65)
def caesar(s,k): return ''.join(sh(c,k) for c in s)
def prog(s,a,b,sign=-1): return ''.join(sh(c, sign*(a*i+b)) for i,c in enumerate(s))
def vig(s,key,sign=-1): return ''.join(sh(c, sign*(ord(key[i%len(key)])-65)) for i,c in enumerate(s))
def cumkey(s,key,sign=-1):
    out=[]; acc=0
    for i,c in enumerate(s):
        acc += ord(key[i%len(key)])-65
        out.append(sh(c, sign*acc))
    return ''.join(out)

keys=['SIGMOID','SIGMOIDS','SIGMOIDX','FOUNDATION','OURFOUNDATION','ALPHABET','HOMEWARD',
 'NAVIGATOR','NAVIGATORS','GREEK','WATCH','COMPANION','COMPANIONS','VOYAGE','VOYAGECIPHER',
 'ISLANDDUEL','TNECI','NETCI','CRUX','SOUTHERNCROSS','STARCHAR','HOMEWARDCHART',
 'THENAVIGATORSHOMEWARDCHART','THEFIVELIGHTS','FIVELIGHTS','LIGHTSOFHOME','KEEPTHEGREEKWATCH',
 'EACHKEEPS','NEARERBEFORE','FOUNDATIONTURNS','UNDOTHOSETURNS','HOMEWARDMESSAGE','MESSAGE']
results=[]
def add(tag,s):
    results.append((score(s), s, tag))

for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (0,1):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        base=f"{od}{'R' if rev else ''}"
        add(base+':plain',seq)
        for k in range(26): add(base+f':caesar{k}',caesar(seq,k))
        for a in range(1,26):
            for b in range(0,26):
                add(base+f":prog{a},{b}",prog(seq,a,b))
                add(base+f':prog+{a},{b}',prog(seq,a,b,1))
        for key in keys:
            add(base+f':vig-{key}',vig(seq,key,-1))
            add(base+f':vig+{key}',vig(seq,key,1))
            add(base+f':cum-{key}',cumkey(seq,key,-1))
            add(base+f':cum+{key}',cumkey(seq,key,1))
        add(base+':rev',seq[::-1])
        # per-group guide shifts
        for sign in (-1,1):
            add(base+f':guide{sign}', ''.join(sh(c,sign*(ord(g)-65)) for g in order for c in (groups[g][::-1] if rev else groups[g])))
            gv={'N':13,'E':5,'T':19,'C':22,'I':9}
            add(base+f':greek{sign}', ''.join(sh(c,sign*gv[g]) for g in order for c in (groups[g][::-1] if rev else groups[g])))

results.sort(reverse=True)
print("total",len(results))
seen=set()
n=0
for sc,s,tag in results:
    if s in seen: continue
    seen.add(s); n+=1
    print(f"{sc:7.3f} {s}  {tag}")
    if n>=60: break
print("--- exact dict hits:",[ (s,tag) for sc,s,tag in results if s in WSET][:20])
