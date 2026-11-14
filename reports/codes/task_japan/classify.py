import numpy as np, json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

X=np.load('/home/kozak/mllm/MML/work/solutions/task_japan/Xcred.npy')
lab=json.load(open('/home/kozak/mllm/MML/work/solutions/task_japan/labels.json'))
keys=json.load(open('/home/kozak/mllm/MML/work/solutions/task_japan/keys.json'))
y=np.array(lab)
loo=LeaveOneOut()
for name,clf in [
  ('logreg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
  ('rf', RandomForestClassifier(n_estimators=500, random_state=0)),
  ('svc', make_pipeline(StandardScaler(), SVC(kernel='rbf', C=3))),
]:
    pred=cross_val_predict(clf, X, y, cv=loo)
    acc=(pred==y).mean()
    print(name, 'LOO acc %.3f'%acc)
    print(confusion_matrix(y,pred,labels=sorted(set(y))))
