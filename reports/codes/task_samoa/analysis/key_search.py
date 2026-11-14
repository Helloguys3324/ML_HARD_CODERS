import itertools, sys
sys.path.insert(0,'analysis')
from model import groups, names, score
words=set(w.strip().upper() for w in open('/tmp/en_words.txt') if w.strip().isalpha())
def sh(c,k): return chr((ord(c)-65+k)%26+65)
def vigen(s,key,sign=-1,off=0):
    return ''.join(sh(c, sign*(ord(key[(i+off)%len(key)])-65)) for i,c in enumerate(s))
keys=['NUEPSILONTAUCHIIOTA','EPSILONIOTANUTAUCHI','NUEPSILONTAUCHIIOTAEXTRA',
 'NU','EPSILON','TAU','CHI','IOTA','NUEPSILONTAUCHIIOTA',
 'ALPHABETAGAMMADELTAEPSILON','ALPHA','BETA','GAMMA','DELTA','EPSILON',
 'ACRUXMIMOSAGACRUXIMAIGINAN','ACRUX','MIMOSA','GACRUX','IMAI','GINAN',
 'SUMU','TRIGGERFISH','TOLOA','LUATAGATA','TAUTAI','FETU','MOANA','VAKA','VA\'A',
 'SIGMOID','SIGMOIDS','FOUNDATION','OURFOUNDATION','THEFIVELIGHTS',
 'THENAVIGATORSHOMEWARDCHART','THENAVIGATOR','HOMEWARDCHART','VOYAGECIPHER',
 'ISLANDDUELVOYAGECIPHER','GREEKWATCH','GREEK','WATCH','SAMOA','APIA',
 'CRUX','SOUTHERNCROSS','SOUTHERNCRUX','NAVIGATOR','WAYFINDER','WAYFINDING',
 'HOMEWARD','MESSAGE','HOMEWARDMESSAGE','ALPHABET','LIGHTHOUSE']
res=[]
for order in itertools.permutations(names):
    od=''.join(order)
    for rev in (False,True):
        seq=''.join(''.join(groups[g][::-1] if rev else groups[g]) for g in order)
        for key in keys:
            for sign in (-1,1):
                for off in range(len(key)):
                    s=vigen(seq,key,sign,off)
                    if s in words:
                        res.append((s,od,rev,key,sign,off))
                    else:
                        sc=score(s)
                        res.append((None,sc,s,od,rev,key,sign,off))
print("dict hits:",[r for r in res if r[0]])
scored=[r for r in res if r[0] is None]
scored.sort(key=lambda x:x[1],reverse=True)
for r in scored[:20]: print(r[1],r[2],r[3],r[4],r[5],r[6],r[7])
