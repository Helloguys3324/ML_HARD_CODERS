import numpy as np, csv, time
from scipy.io import wavfile
import torch
from challenge_model import load_model

torch.set_num_threads(8)
m=load_model()
rows=list(csv.DictReader(open('data/recordings.csv')))
label_map={'birds':0,'frogs':1,'insects':2,'rain':3}
clips=[]
for r in rows:
    rate,smp=wavfile.read('data/'+r['path']); clips.append(smp.astype(np.float32)/32768.0)
audio=np.stack(clips)
true=np.array([label_map[r['label']] for r in rows])
N=32000; sr=16000; TARGET=3
freqs=np.fft.rfftfreq(N,1/sr)
NMS=['birds','frogs','insects','rain']

def run(corner, idxs, steps=50, lr=2.0):
    mask=torch.tensor((freqs<corner).astype(np.float32))
    out=[]
    for idx in idxs:
        x0=torch.tensor(audio[idx][None],dtype=torch.float32)
        xt=x0.clone().requires_grad_(True)
        budget=0.1*x0.norm()
        for i in range(steps):
            p=out_pred=None
            loss=-(m(xt).log_softmax(-1)[:,TARGET].mean())
            g=torch.autograd.grad(loss,xt)[0]
            G=torch.fft.rfft(g,dim=-1)*mask
            gl=torch.fft.irfft(G,n=N,dim=-1)
            xt=xt-lr*gl
            d=xt-x0; dn=d.norm()
            if dn>budget: d=d/dn*budget
            xt=(x0+d).detach().clone().requires_grad_(True)
        with torch.no_grad():
            pr=m(xt).softmax(-1).numpy()[0]
        out.append(f"idx{idx}({NMS[true[idx]]})->{NMS[pr.argmax()]} p3={pr[3]:.3f}")
    print(f"corner<{corner}Hz:", " | ".join(out), flush=True)

run(150, [2,6,3,14,10,25])
run(8000, [2,6,3,14,10,25])