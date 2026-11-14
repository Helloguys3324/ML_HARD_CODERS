import json, numpy as np, os, collections
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

BASE='/home/kozak/mllm/MML/work/solutions/task_japan'
F=json.load(open(BASE+'/features.json'))
S=json.load(open(BASE+'/spectra.json'))

def feats(row, spec):
    v=[]
    q=np.array(row['q'])  # 7x3
    # brightness-normalized chromaticity of percentiles
    for k in range(q.shape[0]):
        r,g,b=q[k]
        s=r+g+b+1e-6
        v += [r/s, g/s, b/s]
    v += row['mean']
    v += row['std']
    v += [row['sat_mean'],row['sat_std']]+row['sat_q']
    v += [row['rg'],row['gb'],row['rb']]
    v += [row['noise_0.6'],row['noise_1.2'],row['noise_2.5'],row['noise_R'],row['noise_G'],row['noise_B']]
    v += [row['grad_mean'],row['grad_p99'],row['lap_var'],row['hf_ratio'],row['y_mean']]
    v += [np.log(row['size'])]
    # spectrum normalized, bins 3..40
    sp=np.array(spec['spec'][:45]); sp=sp/(sp.sum()+1e-9)
    v += sp.tolist()
    return v

X=[]; y=[]; files=[]
for row in F:
    rel=row['rel']
    if rel.startswith('credited'):
        h=rel.split('/')[1]
        # find spec
        spec=S['credited'][h][int(os.path.basename(rel)[5:8])-1] if False else None
        # spectra stored in same order as sorted files
        idx=[r['file'] for r in F if r['rel'].startswith('credited/'+h+'/')].index(row['file'])
        spec=S['credited'][h][idx]
        X.append(feats(row,spec)); y.append(h); files.append(rel)
    else:
        spec=S['anon'][os.path.basename(rel)]
        X.append(feats(row,spec)); y.append('anonymous'); files.append(rel)
X=np.array(X); y=np.array(y)
mask=y!='anonymous'
Xc=X[mask]; yc=y[mask]
print('cred shape',Xc.shape)
for name,clf in [('logreg',make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.3))),
                 ('rf',RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=1))]:
    pred=cross_val_predict(clf,Xc,yc,cv=LeaveOneOut())
    print(name,'LOO acc %.3f'%(pred==yc).mean())
    print(confusion_matrix(yc,pred,labels=sorted(set(yc))))
np.save(BASE+'/Xfull.npy',X); np.save(BASE+'/yfull.npy',y)
json.dump(files,open(BASE+'/filesfull.json','w'))
