import json, numpy as np, collections, os

F=json.load(open('/home/kozak/mllm/MML/work/solutions/task_japan/features.json'))

def get(row, key):
    v=row.get(key)
    return v

# build numeric matrix
keys=[]
for k,v in F[0].items():
    if k in ('file','rel','line_bboxes','text_top','q'): continue
    if isinstance(v,(int,float)): keys.append(k)
keys += ['mean0','mean1','mean2','std0','std1','std2','rg','gb','rb']

def mat(row):
    out=[]
    for k in keys:
        if k.startswith('mean') and k[-1] in '012':
            out.append(row['mean'][int(k[-1])])
        elif k.startswith('std') and k[-1] in '012':
            out.append(row['std'][int(k[-1])])
        else:
            out.append(row.get(k))
    return out

def ordered():
    cred=[]; anon=[]
    for r in F:
        if r['rel'].startswith('credited'): cred.append(r)
        else: anon.append(r)
    return cred, anon

cred, anon = ordered()
labels=[r['rel'].split('/')[1] for r in cred]
X=np.array([mat(r) for r in cred],dtype=float)
Xa=np.array([mat(r) for r in anon],dtype=float)
print('features:',len(keys))

# ANOVA F per feature
import itertools
def fstat(col, lab):
    groups=[col[np.array(lab)==g] for g in sorted(set(lab))]
    gm=col.mean()
    between=sum(len(g)*(g.mean()-gm)**2 for g in groups)/(len(groups)-1)
    within=sum(((g-g.mean())**2).sum() for g in groups)/(len(col)-len(groups))
    return between/(within+1e-12)
rows=[]
for j,k in enumerate(keys):
    col=X[:,j]
    if np.std(col)==0: continue
    f=fstat(col,labels)
    rows.append((f,k,col.mean(),col.std()))
rows.sort(reverse=True)
print('Top ANOVA F features (credited, 6 groups):')
for f,k,m,s in rows[:25]:
    print('  %-12s F=%7.2f  mean %.3f std %.3f'%(k,f,m,s))

# means per group for top features
print('\nPer-person means for top 12 features:')
for f,k,m,s in rows[:12]:
    j=keys.index(k)
    line='  %-12s'%k
    for g in sorted(set(labels)):
        line+=' %8.2f'%X[np.array(labels)==g,j].mean()
    print(line)

np.save('/home/kozak/mllm/MML/work/solutions/task_japan/Xcred.npy',X)
np.save('/home/kozak/mllm/MML/work/solutions/task_japan/Xanon.npy',Xa)
json.dump(keys, open('/home/kozak/mllm/MML/work/solutions/task_japan/keys.json','w'))
json.dump(labels, open('/home/kozak/mllm/MML/work/solutions/task_japan/labels.json','w'))
json.dump([r['file'] for r in anon], open('/home/kozak/mllm/MML/work/solutions/task_japan/anon_files.json','w'))
