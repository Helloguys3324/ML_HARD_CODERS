import pandas as pd, numpy as np, torch, torch.nn as nn, time, json, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(20)

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy().astype(np.float32); y = train.vrancea.to_numpy().astype(np.float32)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))


def make_mlp(nin, hidden=256, layers=3, dropout=0.3):
    mods = []
    prev = nin
    for _ in range(layers):
        mods += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        prev = hidden
    mods += [nn.Linear(prev, 1)]
    return nn.Sequential(*mods)


def train_model(model, Xtr, ytr, Xva, epochs=120, bs=256, lr=1e-3, wd=1e-4):
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(1)
    Xva_t = torch.tensor(Xva)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xva_t).squeeze(1).numpy()
    return 1/(1+np.exp(-logits))


def run_nn(name, build, qt=True):
    t0 = time.time()
    accs, aucs = [], []
    for tr, va in folds:
        if qt:
            q = QuantileTransformer(output_distribution='normal', random_state=0)
            Xtr = q.fit_transform(X[tr]).astype(np.float32); Xva = q.transform(X[va]).astype(np.float32)
        else:
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]).astype(np.float32); Xva = sc.transform(X[va]).astype(np.float32)
        model = build(48)
        p = train_model(model, Xtr, y[tr], Xva)
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int)))
        aucs.append(roc_auc_score(y[va], p))
    print(f'{name:18s} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} std={np.std(accs):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return np.mean(accs), np.mean(aucs)


if __name__ == '__main__':
    res = {}
    for name, build in [
        ('MLP256x3', lambda n: make_mlp(n, 256, 3, 0.3)),
        ('MLP512x3', lambda n: make_mlp(n, 512, 3, 0.3)),
        ('MLP128x4', lambda n: make_mlp(n, 128, 4, 0.4)),
    ]:
        a, u = run_nn(name, build)
        res[name] = {'acc': a, 'auc': u}
    print('SUMMARY', json.dumps(res), flush=True)