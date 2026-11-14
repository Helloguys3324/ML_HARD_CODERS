"""Image duplicate-detection utilities: perceptual hashes, feature matching, verification."""
import numpy as np
import cv2


def load_gray(path, max_dim=512):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    return im


def load_rgb(path, max_dim=512):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA)
    return im


def dhash(gray, hash_size=16):
    """Difference hash. Returns int64."""
    g = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    return _array_to_int(diff)


def ahash(gray, hash_size=16):
    g = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = g.mean()
    return _array_to_int(g > mean)


def phash(gray, hash_size=32, highfreq_factor=4):
    """DCT perceptual hash (pHASH)."""
    g = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)
    dctlowfreq = dct[:hash_size // highfreq_factor, :hash_size // highfreq_factor]
    med = np.median(dctlowfreq)
    return _array_to_int(dctlowfreq > med)


def _array_to_int(a):
    flat = a.flatten()
    val = 0
    for bit in flat:
        val = (val << 1) | (1 if bit else 0)
    return val


def hamming(a, b):
    x = int(a) ^ int(b)
    return bin(x).count('1')


def color_hist(rgb, bins=8):
    h = cv2.calcHist([rgb], [0, 1, 2], None, [bins, bins, bins],
                     [0, 256, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h.flatten().astype(np.float32)


def rgb_mean(rgb):
    return rgb.mean(axis=(0, 1)).astype(np.float32)


def sift_keypoints(gray, max_dim=512):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def orb_keypoints(gray, max_dim=512):
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_homography(kp1, des1, kp2, des2, ratio=0.75):
    """SIFT+FLANN+RANSAC homography. Returns (inliers, H, matches)."""
    if des1 is None or des2 is None:
        return 0, None, 0
    if len(des1) < 4 or len(des2) < 4:
        return 0, None, 0
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        knn = flann.knnMatch(des1, des2, k=2)
    except Exception:
        return 0, None, 0
    good = []
    for m in knn:
        if len(m) >= 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return 0, None, 0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return 0, None, 0
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, H, len(good)


def aligned_similarity(gray1, gray2, H1_to_2):
    """Warp gray2 into gray1's frame and compute SSIM + photometric error on overlap."""
    from skimage.metrics import structural_similarity as ssim
    h, w = gray1.shape
    warped = cv2.warpPerspective(gray2, H1_to_2, (w, h))
    # overlap mask
    mask = cv2.warpPerspective(np.ones_like(gray2, dtype=np.uint8) * 255,
                               H1_to_2, (w, h)) > 127
    if mask.sum() < 1000:
        return None
    a = gray1.astype(np.float32)
    b = warped.astype(np.float32)
    # only evaluate on overlap
    a_m = a[mask]
    b_m = b[mask]
    mae = np.mean(np.abs(a_m - b_m))
    # SSIM on bounding-box overlap
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    patch1 = gray1[y0:y1 + 1, x0:x1 + 1]
    patch2 = warped[y0:y1 + 1, x0:x1 + 1]
    try:
        s = ssim(patch1, patch2, data_range=255)
    except Exception:
        s = 0.0
    return dict(mae=float(mae), ssim=float(s), overlap_px=int(mask.sum()))


def dct_lowfreq(gray, n=6):
    """Low-frequency DCT coefficients as a small translation-tolerant descriptor."""
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)[:n, :n].flatten()
    return dct