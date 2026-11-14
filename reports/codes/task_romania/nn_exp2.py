import pandas as pd, numpy as np, torch, torch.nn as nn, time, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(6)

train = pd.read_csv('challenges/task_romania/data/dataset/train.csv')
CH = [f'c{j+1:02d}' for j in range(48)]
X = train[CH].to_numpy().astype(np.float32); y = train.vrancea.to_numpy().astype(np.float32)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))


class MLP(nn.Module):
    def __init__(self, nin, hidden, layers, dropout, bn=True):
        super().__init__()
        mods = []
        prev = nin
        for _ in range(layers):
            mods.append(nn.Linear(prev, hidden))
            if bn: mods.append(nn.BatchNorm1d(hidden))
            mods.append(nn.ReLU())
            mods.append(nn.Dropout(dropout))
            prev = hidden
        mods.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, nin, hidden, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden*nin, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))
    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


def train_predict(model, Xtr, ytr, Xva, epochs=150, bs=512, lr=2e-3, wd=1e-4):
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(1); Xva_t = torch.tensor(Xva)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return 1/(1+torch.exp(-model(Xva_t).squeeze(1))).numpy()


def run(name, build, is_cnn=False):
    t0 = time.time(); accs, aucs = [], []
    for tr, va in folds:
        qt = QuantileTransformer(output_distribution='normal', random_state=0)
        Xtr = qt.fit_transform(X[tr]).astype(np.float32); Xva = qt.transform(X[va]).astype(np.float32)
        model = build(48)
        p = train_predict(model, Xtr, y[tr], Xva)
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int)))
        aucs.append(roc_auc_score(y[va], p))
    print(f'{name:20s} acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f} std={np.std(accs):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return float(np.mean(accs)), float(np.mean(aucs))


if __name__ == '__main__':
    res = {}
    for name, build in [
        ('MLP-256x3', lambda n: MLP(n, 256, 3, 0.3)),
        ('MLP-512x4', lambda n: MLP(n, 512, 4, 0.35)),
        ('CNN1D-64', lambda n: CNN1D(n, 64, 0.3)),
    ]:
        a, u = run(name, build)
        res[name] = {'acc': a, 'auc': u}
    print('SUMMARY', json.dumps(res), flush=True)