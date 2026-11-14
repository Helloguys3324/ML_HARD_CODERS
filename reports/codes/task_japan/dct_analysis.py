import json, numpy as np, os, collections

D=json.load(open('/home/kozak/mllm/MML/work/solutions/task_japan/dct.json'))

def periodicity(h):
    h=np.array(h,dtype=float)
    h=h-h.mean()
    # FFT of histogram (length 121, values -60..60)
    F=np.abs(np.fft.rfft(h))
    # periods 2..20 correspond to frequencies 121/p
    out={}
    for p in range(2,21):
        k=int(round(121/p))
        if 0<k<len(F): out[p]=F[k]
    return out

people=sorted(set(v['rel'].split('/')[1] for v in D.values() if v['rel'].startswith('credited')))
freqs=['dct_1_1','dct_2_1','dct_2_2','dct_3_3','dct_4_4']
for f in freqs:
    print('='*20,f)
    for h in people:
        rows=[v for v in D.values() if v['rel'].startswith('credited/'+h+'/')]
        P=[periodicity(r['hist'][f]) for r in rows]
        avg={p:np.mean([x[p] for x in P]) for p in range(2,13)}
        # normalize by low freq power
        tot=np.mean([np.sum(np.array(r['hist'][f],float)**2) for r in rows])
        print('%-12s'%h, ' '.join('%d:%.2f'%(p,avg[p]/(tot**.5+1e-9)) for p in range(2,13)))
    an=[v for v in D.values() if v['rel'].startswith('anonymous')]
    P=[periodicity(r['hist'][f]) for r in an]
    avg={p:np.mean([x[p] for x in P]) for p in range(2,13)}
    tot=np.mean([np.sum(np.array(r['hist'][f],float)**2) for r in an])
    print('%-12s'%'anon', ' '.join('%d:%.2f'%(p,avg[p]/(tot**.5+1e-9)) for p in range(2,13)))
