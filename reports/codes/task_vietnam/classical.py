import csv, time, numpy as np
from pathlib import Path
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

root=Path('../../tasks/task_vietnam/data')
rows=[(r['filename'],r['label']) for r in csv.DictReader(open(root/'train-output.csv'))]
rows=[r for r in rows if (root/'train-input'/r[0]).exists()]
labs=sorted(set(l for _,l in rows)); li={l:i for i,l in enumerate(labs)}
print('n',len(rows),labs,flush=True)
t=time.time()
X=[]; y=[]
for f,l in rows:
    im=np.asarray(Image.open(root/'train-input'/f).convert('RGB').resize((128,128)))
    g=rgb2gray(im)
    h=hog(g,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)
    # color histogram 4x4 spatial (RGB, 8 bins)
    q=(im//32).reshape(4,32,4,32,3)
    hist=np.zeros((4,4,8,3),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            for c in range(3):
                hist[i,j,:,c]=np.bincount(q[i,:,j,:,c].ravel(),minlength=8)
    hist=hist.ravel(); hist/= (hist.sum()+1e-8)
    lbp=local_binary_pattern((g*255).astype(np.uint8),8,1,method='uniform')
    lb=np.bincount(lbp.astype(int).ravel(),minlength=10).astype(np.float32); lb/=lb.sum()
    X.append(np.concatenate([h,hist,lb])); y.append(li[l])
X=np.stack(X); y=np.array(y)
print('features',X.shape,'time',time.time()-t,flush=True)
np.savez_compressed('/tmp/vn_classical.npz',X=X,y=y,rows=np.array([r[0] for r in rows]))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for name,clf in [('svc-rbf',SVC(C=10,gamma='scale',probability=True)),('svc-lin',LinearSVC(C=1)),('logreg',LogisticRegression(max_iter=2000,C=1))]:
    t=time.time(); s=cross_val_score(clf,X,y,cv=cv,n_jobs=1); print(name,round(s.mean(),4),s.round(4),'time',round(time.time()-t,1),flush=True)
