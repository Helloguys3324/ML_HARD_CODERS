"""Extract JPEG structure fingerprints: DQT tables, DHT tables, subsampling, encoder markers."""
import sys, glob, os, struct, hashlib, collections, json

def parse_jpeg(path):
    data = open(path, 'rb').read()
    i = 2
    out = {'path': path, 'size': len(data), 'markers': [], 'dqt': [], 'dht': [], 'sof': None, 'app': []}
    while i < len(data) - 1:
        if data[i] != 0xFF:
            # scan for next marker in entropy data
            j = data.find(b'\xff', i)
            if j < 0: break
            i = j
            continue
        m = data[i+1]
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2; continue
        if m == 0xFF:
            i += 1; continue
        if m == 0xDA:
            L = struct.unpack('>H', data[i+2:i+4])[0]
            out['markers'].append('SOS')
            # skip entropy data to next real marker
            j = i + 2 + L
            while j < len(data)-1:
                if data[j] == 0xFF:
                    mm = data[j+1]
                    if mm == 0x00 or 0xD0 <= mm <= 0xD7:
                        j += 2; continue
                    break
                j += 1
            i = j
            continue
        L = struct.unpack('>H', data[i+2:i+4])[0]
        seg = data[i+4:i+2+L]
        name = 'APP%d' % (m-0xE0) if 0xE0 <= m <= 0xEF else ('SOF%d' % (m-0xC0) if 0xC0 <= m <= 0xCF else hex(m))
        out['markers'].append(name)
        if 0xE0 <= m <= 0xEF:
            out['app'].append((name, seg[:16].hex(), len(seg)))
        if m == 0xDB:
            # quantization table(s)
            p = 0
            while p < len(seg):
                pq = seg[p] >> 4; tq = seg[p] & 15; p += 1
                n = 64 * (2 if pq else 1)
                tbl = seg[p:p+n]; p += n
                out['dqt'].append((pq, tq, hashlib.md5(tbl).hexdigest()[:8], tbl.hex()))
        if m == 0xC4:
            p = 0
            while p < len(seg):
                tc = seg[p] >> 4; th = seg[p] & 15; p += 1
                counts = seg[p:p+16]; p += 16
                n = sum(counts); p += n
                out['dht'].append((tc, th, hashlib.md5(seg[p-n-16:p]).hexdigest()[:8]))
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            prec = seg[0]; h, w = struct.unpack('>HH', seg[1:5]); nc = seg[5]
            comps = []
            for c in range(nc):
                cid, hv, tq = seg[6+3*c], seg[7+3*c], seg[8+3*c]
                comps.append((cid, hv >> 4, hv & 15, tq))
            out['sof'] = (m, prec, w, h, nc, tuple(comps))
        i += 2 + L
    return out

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    files = sorted(glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True))
    sigs = collections.defaultdict(list)
    for f in files:
        r = parse_jpeg(f)
        key = (json.dumps(r['dqt']), json.dumps(r['dht']), str(r['sof']), str(r['markers']), str(r['app']))
        sigs[key].append(f)
    print('files:', len(files), 'unique structural sigs:', len(sigs))
    for k, v in sigs.items():
        print('---', len(v), 'files')
        print(dir and k[0][:1] and '')
        print(k[0][:300])
        print(k[1][:300], '|', k[2], '|', k[3], '|', k[4])
        print('  examples:', [os.path.relpath(x, root) for x in v[:5]])
