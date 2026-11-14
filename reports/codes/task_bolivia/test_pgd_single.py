import numpy as np, csv
from scipy.io import wavfile
import torch
from challenge_model import load_model

m = load_model()
torch.set_num_threads(8)
rows = list(csv.DictReader(open('data/recordings.csv')))
label_map = {'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000

# pick a correct non-rain recording to attack (index 2 birds, 3 frogs, 10 insects, 25 insects)
TARGET=3
def targeted_pgd(x0, steps=40, lr=0.02):
    x = torch.tensor(x0[None], dtype=torch.float32)
    x = x.clone().requires_grad_(True)
    x_norm = x.norm()
    budget = 0.1 * x_norm  # sqrt(0.01)*||x||
    for i in range(steps):
        out = m(x)
        loss = -out.log_softmax(-1)[:,TARGET].mean()
        # also keep rain confident
        g = torch.autograd.grad(loss, x)[0]
        x = x - lr * g
        # project to energy ball
        d = x - torch.tensor(x0[None], dtype=torch.float32)
        dn = d.norm()
        if dn > budget:
            d = d / dn * budget
        x = (torch.tensor(x0[None],dtype=torch.float32) + d).detach().clone().requires_grad_(True)
    with torch.no_grad():
        return m(x).softmax(-1).numpy()[0]

for idx in [2,3,10,25,6,14]:
    p = targeted_pgd(audio[idx])
    print(f"idx {idx} true={["birds","frogs","insects","rain"][true[idx]]} after-PGD pred={["birds","frogs","insects","rain"][p.argmax()]} p={p}")