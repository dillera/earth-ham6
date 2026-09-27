# Earth — OCS HAM6 backdrop (equirectangular)

Source: NASA Blue Marble (cloudless, public domain), equirectangular: lon -180..+180 left→right, lat +90..-90 top→bottom.

| File | What |
|---|---|
| `earth_ham6_pal.iff`  | 320x256 PAL lores HAM6 IFF ILBM (ByteRun1, CAMG $21800) — loads in DPaint/PPaint/ViewTek/MultiView |
| `earth_ham6_ntsc.iff` | 320x200 NTSC lores HAM6 IFF ILBM (CAMG $11800) |
| `*.raw` | 6 bitplanes, non-interleaved, 40 bytes/row (plane0 first). 61440 B PAL / 48000 B NTSC |
| `*.pal` | 16 x UWORD $0RGB, ready to poke into COLOR00–COLOR15 |
| `*_preview.png` | exact decode of what the Amiga will show |
| `ham6.py` | encoder (optimal per-line Viterbi HAM search) — `python3 ham6.py src2048.png` |

COLOR00 is black (border / HAM line start). Display: BPLCON0 = $6A00 (6 planes + HAM + color), BPL1MOD/BPL2MOD = 0.

## ISS lat/lon → pixel
    x = (lon + 180) * W / 360          ; W = 320
    y = (90 - lat)  * H / 180          ; H = 256 (PAL) or 200 (NTSC)

Tip: use hardware sprites for the ISS marker and ground track — sprites are unaffected by HAM, whereas blitting into HAM playfields causes colour fringing.
