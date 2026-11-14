import itertools, sys, math
sys.path.insert(0,'analysis')
from model import groups, names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
def sh(c,k): return chr((ord(c)-65+k)%26+65)
def shs(s,k): return ''.join(sh(c,k) for c in s)
def rail_dec(s,r):
    n=len(s); pattern=[]; row=0; d=1
    for i in range(n):
        pattern.append(row)
        if row==0: d=1
        elif row==r-1: d=-1
        row+=d
    counts=[pattern.count(k) for k in range(r)]
    rows=[]; idx=0
    for k in range(r):
        rows.append(list(s[idx:idx+counts[k]])); idx+=counts[k]
    out=[]; ptr=[0]*r
    for rr in pattern:
        out.append(rows[rr][ptr[rr]]); ptr[rr]+=1
    return ''.join(out)
def col_dec(s,cols,direction=0):
    n=len(s); rows=(n+cols-1)//cols
    grid=[['']*cols for _ in range(rows)]
    if direction==0:
        it=iter(s)
        for c in range(cols):
            for r in range(rows):
                if r*cols+c<n: grid[r][c]=next(it)
        return ''.join(grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c])
    else:
        it=iter(s)
        for c in range(cols):
            for r in range(rows):
                if r*cols+c<n: grid[r][c]=next(it)
        return ''.join(grid[r][c] for c in range(cols) for r in range(rows) if grid[r][c])
res=[]; hits=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        variants={'plain':seq,'rev':seq[::-1]}
        for r in range(2,6):
            variants[f'rail{r}']=rail_dec(seq,r)
            variants[f'rail{r}R']=rail_dec(seq[::-1],r)
        for c in range(2,7):
            variants[f'col{c}']=col_dec(seq,c,0)
            variants[f'col{c}R']=col_dec(seq,c,1)
        for name,s in variants.items():
            for k in range(26):
                t=shs(s,k)
                if t in words: hits.append((t,od,rev,name,k))
                res.append((score(t),t,od,rev,name,k))
res.sort(reverse=True)
print("hits",hits[:30])
for r in res[:10]: print(r)
