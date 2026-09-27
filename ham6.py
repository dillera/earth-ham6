#!/usr/bin/env python3
"""OCS HAM6 encoder.
 - optimal per-scanline Viterbi search over all 4096 OCS colours
 - vertical error diffusion between scanlines
 - iterative refinement of the 16-colour base palette (COLOR00 fixed black)
Writes IFF ILBM (ByteRun1), raw planar bitplanes, palette, and a decoded preview PNG."""
import struct, sys
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

W_ERR = np.array([0.30, 0.59, 0.11]) * 3  # luma-weighted RGB error
R, Gc, B = np.meshgrid(np.arange(16), np.arange(16), np.arange(16), indexing="ij")
GRID = np.stack([R, Gc, B], -1).astype(float)


def load(src, w, h):
    im = Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)
    im = ImageEnhance.Color(im).enhance(1.10)
    im = im.filter(ImageFilter.UnsharpMask(radius=0.8, percent=50, threshold=2))
    return np.asarray(im, dtype=np.float64) / 17.0  # 0..15 in 12-bit units


def kmeans(px, k, fixed, iters=30, seed=1):
    rng = np.random.default_rng(seed)
    c = np.vstack([fixed, px[rng.choice(len(px), k - len(fixed), replace=False)]])
    for _ in range(iters):
        a = ((((px[:, None, :] - c[None]) ** 2) * W_ERR).sum(-1)).argmin(1)
        for i in range(len(fixed), k):
            m = px[a == i]
            c[i] = m.mean(0) if len(m) else px[rng.integers(len(px))]
    return np.clip(np.rint(c), 0, 15).astype(int)


def encode_line(row, pal):
    n = len(row)
    pal_idx = pal[:, 0] * 256 + pal[:, 1] * 16 + pal[:, 2]
    prev = np.full(4096, 1e18)
    prev[pal_idx[0]] = 0.0            # HAM line starts from COLOR00
    backs, bestprev = [], []
    for x in range(n):
        p3 = prev.reshape(16, 16, 16)
        mR, aR = p3.min(0), p3.argmin(0)
        mG, aG = p3.min(1), p3.argmin(1)
        mB, aB = p3.min(2), p3.argmin(2)
        cand = np.stack([np.broadcast_to(mR[None], (16, 16, 16)),
                         np.broadcast_to(mG[:, None], (16, 16, 16)),
                         np.broadcast_to(mB[:, :, None], (16, 16, 16))])
        ch = cand.argmin(0)
        best = cand.min(0).reshape(-1).copy()
        src = np.where(ch == 0, aR[Gc, B] * 256 + Gc * 16 + B,
              np.where(ch == 1, R * 256 + aG[R, B] * 16 + B,
                                R * 256 + Gc * 16 + aB[R, Gc])).reshape(-1).astype(np.int32)
        bi = int(prev.argmin()); bv = prev[bi]
        for i, pi in enumerate(pal_idx):
            if bv < best[pi]:
                best[pi] = bv; src[pi] = -1 - i
        backs.append(src); bestprev.append(bi)
        err = (((GRID - row[x]) ** 2) * W_ERR).sum(-1).reshape(-1)
        prev = best + err
    s = int(prev.argmin())
    codes = np.zeros(n, np.uint8); cols = np.zeros((n, 3), int)
    for x in range(n - 1, -1, -1):
        r, g, b = s >> 8, (s >> 4) & 15, s & 15
        cols[x] = (r, g, b)
        p = int(backs[x][s])
        if p < 0:
            codes[x] = -1 - p; s = bestprev[x]
        else:
            if p >> 8 != r: codes[x] = 0x20 | r            # 10xxxx modify red
            elif (p >> 4) & 15 != g: codes[x] = 0x30 | g   # 11xxxx modify green
            else: codes[x] = 0x10 | b                      # 01xxxx modify blue / hold
            s = p
    return codes, cols


def encode(img, pal, diffuse=0.55, step=1):
    h, w, _ = img.shape
    codes = np.zeros((h, w), np.uint8); out = np.zeros((h, w, 3), int)
    carry = np.zeros((w, 3))
    for y in range(0, h, step):
        tgt = np.clip(img[y] + carry, 0, 15)
        c, col = encode_line(tgt, pal)
        codes[y] = c; out[y] = col
        e = tgt - col
        e[np.abs(e) < 0.5] = 0          # dead zone: keep flat areas (ocean) solid
        carry = e * diffuse if step == 1 else 0
    return codes, out


def decode(codes, pal):  # independent HAM6 decoder for verification
    h, w = codes.shape
    out = np.zeros((h, w, 3), int)
    for y in range(h):
        cur = pal[0].copy()
        for x in range(w):
            v = int(codes[y, x]); m, d = v >> 4, v & 15
            if m == 0: cur = pal[d].copy()
            elif m == 1: cur[2] = d
            elif m == 2: cur[0] = d
            else: cur[1] = d
            out[y, x] = cur
    return out


def planes(codes):
    h, w = codes.shape
    return [np.packbits(((codes >> p) & 1).astype(np.uint8), axis=1) for p in range(6)]  # each h x w/8


def byterun1(row):
    out = bytearray(); i = 0; n = len(row)
    while i < n:
        j = i
        while j < n - 1 and row[j] == row[j + 1] and j - i < 127: j += 1
        if j > i + 1:
            out += bytes([257 - (j - i + 1)]) + bytes([row[i]]); i = j + 1
        else:
            j = i
            while j < n and j - i < 128 and not (j < n - 2 and row[j] == row[j + 1] == row[j + 2]): j += 1
            j = max(j, i + 1)
            out += bytes([j - i - 1]) + bytes(row[i:j]); i = j
    return bytes(out)


def chunk(tag, data):
    d = tag.encode() + struct.pack(">I", len(data)) + data
    return d + (b"\0" if len(data) & 1 else b"")


def write_ilbm(path, codes, pal, camg, yaspect):
    h, w = codes.shape
    pl = planes(codes)
    bmhd = struct.pack(">HHhhBBBBHBBhh", w, h, 0, 0, 6, 0, 1, 0, 0, 10, yaspect, w, h)
    cmap = bytes(int(v) * 17 for c in pal for v in c)
    body = b"".join(byterun1(pl[p][y]) for y in range(h) for p in range(6))
    form = b"ILBM" + chunk("BMHD", bmhd) + chunk("CMAP", cmap) + chunk("CAMG", struct.pack(">I", camg)) + chunk("BODY", body)
    open(path, "wb").write(b"FORM" + struct.pack(">I", len(form)) + form)


def main(src, w, h, name, camg, yaspect, aspect_preview):
    img = load(src, w, h)
    px = img.reshape(-1, 3)[::7]
    pal = kmeans(px, 16, fixed=np.array([[0.0, 0.0, 0.0]]))
    for it in range(4):  # refine palette against what HAM actually uses for palette loads
        codes, out = encode(img, pal, step=4)
        rows = np.arange(0, h, 4)
        cs, tg = codes[rows], img[rows]
        for i in range(1, 16):
            m = cs == i
            if m.sum() > 3: pal[i] = np.clip(np.rint(tg[m].mean(0)), 0, 15)
        e = (((out[rows] - tg) ** 2) * W_ERR).sum(-1).mean()
        print(f"  iter {it}: mse={e:.4f}", file=sys.stderr)
    codes, out = encode(img, pal)
    dec = decode(codes, pal)
    assert (dec == out).all(), "decoder mismatch"
    write_ilbm(f"{name}.iff", codes, pal, camg, yaspect)
    pl = planes(codes)
    open(f"{name}.raw", "wb").write(b"".join(p.tobytes() for p in pl))  # 6 consecutive planes
    open(f"{name}.pal", "wb").write(b"".join(struct.pack(">H", (r << 8) | (g << 4) | b) for r, g, b in pal))
    prev = Image.fromarray((dec * 17).astype(np.uint8))
    prev.save(f"{name}_preview.png")
    prev.resize((w * 3, int(h * 3 * aspect_preview)), Image.NEAREST).save(f"{name}_preview_3x.png")
    print(name, "palette:", " ".join(f"${r:X}{g:X}{b:X}" for r, g, b in pal))


if __name__ == "__main__":
    src = sys.argv[1]
    main(src, 320, 256, "earth_ham6_pal", 0x00021800, 11, 1.0)    # PAL lores HAM
    main(src, 320, 200, "earth_ham6_ntsc", 0x00011800, 10, 1.0)   # NTSC lores HAM
