import json, numpy as np, os, itertools, collections

D=json.load(open('/home/kozak/mllm/MML/work/solutions/task_japan/textlines.json'))

def sig(rec):
    lines=rec['lines']
    # filter: lines that are horizontal text (height 12..40, width>100)
    ls=[l for l in lines if 12<=l['h']<=40 and l['right']-l['left']>80]
    if len(ls)<2: return None
    h=np.median([l['h'] for l in ls])
    vals=[]
    for l in ls:
        vals.append((l['right']-l['left'])/h)
    return np.array(vals)

def dist(a,b):
    if a is None or b is None or len(a)!=len(b): return None
    return float(np.linalg.norm(a-b)/np.sqrt(len(a)))

groups=collections.defaultdict(list)
for rel,rec in D.items():
    s=sig(rec)
    if s is not None:
        g=rel.split('/')[1] if rel.startswith('credited') else 'anonymous'
        groups[g].append((rel,s))

print('images with signature:', {k:len(v) for k,v in groups.items()})
# pairwise within vs between for credited
within=[]; between=[]
people=[k for k in groups if k!='anonymous']
for p in people:
    for (r1,s1),(r2,s2) in itertools.combinations(groups[p],2):
        d=dist(s1,s2)
        if d is not None: within.append(d)
for p,q in itertools.combinations(people,2):
    for r1,s1 in groups[p]:
        for r2,s2 in groups[q]:
            d=dist(s1,s2)
            if d is not None: between.append(d)
print('within-person caption dist: mean %.3f median %.3f n=%d'%(np.mean(within),np.median(within),len(within)))
print('between-person caption dist: mean %.3f median %.3f n=%d'%(np.mean(between),np.median(between),len(between)))
# closest pairs across all
allitems=[(r,s) for v in groups.values() for r,s in v]
best=[]
for i in range(len(allitems)):
    for j in range(i+1,len(allitems)):
        d=dist(allitems[i][1],allitems[j][1])
        if d is not None and d<0.25:
            best.append((d,allitems[i][0],allitems[j][0]))
best.sort()
print('\nclosest caption matches:')
for d,a,b in best[:25]:
    print(' %.3f %s %s'%(d,a,b))
print('num close pairs', len(best))
