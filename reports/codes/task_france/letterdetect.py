import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# ---- letter templates via PIL ----
def make_templates(chars, font_px=32, pad=2):
    """Return dict char -> (mask_2d, size). Letters rendered at font_px height."""
    out = {}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_px)
    for ch in chars:
        # measure
        tmp = Image.new('L', (font_px*2, font_px*2), 0)
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0,0), ch, font=font)
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        img = Image.new('L', (font_px*2, font_px*2), 0)
        d = ImageDraw.Draw(img)
        d.text((font_px - (bbox[0]+bbox[2])/2, font_px - (bbox[1]+bbox[3])/2), ch, fill=255, font=font)
        a = np.array(img) > 128
        # crop to content
        ys, xs = np.where(a)
        a = a[ys.min():ys.max()+1, xs.min():xs.max()+1]
        out[ch] = (a, w, h)
    return out

def letter_match(bitmap, templates, max_scale=4.0, min_overlap=0.35):
    """Slide templates over a binary bitmap, return list of (char, score, x_center, y_center, scale).
    bitmap: 2D bool array, True = point present."""
    H, W = bitmap.shape
    results = []
    # pyramid: test multiple scales of bitmap
    scales = np.arange(0.6, max_scale+0.01, 0.2)
    for ch, (amask, w0, h0) in templates.items():
        th, tw = amask.shape
        # test at each global scale (shrink bitmap, keep template fixed -> equivalent to scaling template up)
        for s in scales:
            # effective: template covers th*s, tw*s pixels in bitmap
            if th*s > H or tw*s > W:
                continue
            # downscale bitmap by s so template matches at native size
            if s >= 1.0:
                nH, nW = int(H/s), int(W/s)
                if nH < th or nW < tw: continue
                # subsample
                small = bitmap.reshape(nH, s, nW, s).any(axis=(1,3)) if False else _subsample(bitmap, nH, nW)
            else:
                nH, nW = int(H/s), int(W/s)
                small = _subsample(bitmap, nH, nW)
            if small.shape[0] < th or small.shape[1] < tw:
                continue
            # correlate count of template pixels that align with a dot
            tm = amask.sum()
            corr = ndimage.correlate(small.astype(float), amask.astype(float), mode='constant')
            score = corr / tm  # fraction of template pixels covered by dots
            best = score.max()
            if best >= min_overlap and best >= 0.3:
                y, x = np.unravel_index(score.argmax(), score.shape)
                # map back to original coordinates
                cy = (y + th/2) * s
                cx = (x + tw/2) * s
                results.append((ch, float(best), float(cx), float(cy), float(s)))
    return results

def _subsample(a, nH, nW):
    """boolean OR subsample to nH x nW."""
    H, W = a.shape
    # integer binning OR
    out = np.zeros((nH, nW), dtype=bool)
    for i in range(nH):
        y0 = int(i*H/nH); y1 = int((i+1)*H/nH)
        for j in range(nW):
            x0 = int(j*W/nW); x1 = int((j+1)*W/nW)
            out[i,j] = a[y0:y1, x0:x1].any()
    return out

def points_to_bitmap(lon, lat, W=400, H=200, xlim=None, ylim=None):
    if xlim is None: xlim = (lon.min(), lon.max())
    if ylim is None: ylim = (lat.min(), lat.max())
    xb = np.linspace(xlim[0], xlim[1], W+1)
    yb = np.linspace(ylim[0], ylim[1], H+1)
    h,_,_ = np.histogram2d(lon, lat, bins=[xb,yb])
    return (h.T > 0)

if __name__ == '__main__':
    templates = make_templates('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print('made templates', len(templates))