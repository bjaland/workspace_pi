"""Anima los gráficos del dashboard GPS en la maqueta del iMac.

Bucle de 10 s:
  0.0-2.4  los gráficos se construyen desde cero (barras, donuts, KPIs, árbol)
  4.2-5.6  actualización de datos en vivo (barras, ranking y KPIs cambian)
  7.0-8.4  vuelven a los valores originales
  9.0-10   se recogen a cero para enlazar con el inicio

Uso:
  python3 animar_imac.py           ->  gps_imac.mp4 / .gif       (imac_original.jpg, 2000x1333)
  python3 animar_imac.py 1080      ->  gps_imac_1080.mp4 / .gif  (imac1080_original.jpg, 1920x1080)
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
# Maquetas: las coordenadas del script están medidas sobre imac_original.jpg;
# (dx, dy) es el desplazamiento del dashboard en cada maqueta respecto a esa.
PROFILES = {
    "2000": dict(src="imac_original.jpg", out="gps_imac", dx=0, dy=0),
    "1080": dict(src="imac1080_original.jpg", out="gps_imac_1080", dx=4, dy=-48),
}
PROF = PROFILES[sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PROFILES else "2000"]
FPS, DUR = 30, 10.0
NF = int(FPS * DUR)
S = 3  # superescalado del área del dashboard para dibujar texto suave

photo = cv2.imread(os.path.join(HERE, PROF["src"]))[:, :, ::-1].copy()
IH, IW = photo.shape[:2]
DX, DY = PROF["dx"], PROF["dy"]
# foto llevada a las coordenadas de referencia
orig = cv2.warpAffine(photo, np.float32([[1, 0, -DX], [0, 1, -DY]]), (max(IW, 2000), max(IH, 1333)),
                      borderMode=cv2.BORDER_REPLICATE)
RX0, RY0, RX1, RY1 = 650, 225, 1475, 720  # área del dashboard
region = orig[RY0:RY1, RX0:RX1]
big0 = cv2.resize(region, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
small0 = cv2.resize(big0, (RX1 - RX0, RY1 - RY0), interpolation=cv2.INTER_AREA)

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
F_KPI = ImageFont.truetype(FONT, int(10 * S))
F_LAB = ImageFont.truetype(FONT, int(6.5 * S))
F_BAR = ImageFont.truetype(FONT, int(5.5 * S))
BLUE = (22, 86, 160)
TEXT = (80, 80, 88)


def P(x, y):
    """coordenada de la foto -> coordenada del lienzo superescalado"""
    return (int(round((x - RX0) * S)), int(round((y - RY0) * S)))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def euro(v):
    return ("-€ " if v < 0 else "€ ") + f"{abs(int(round(v))):,}"


def col(x, y):
    return tuple(int(c) for c in orig[y, x])


# ---------------------------------------------------------------- elementos
KPI = [  # centro x, y, caja a borrar, valor
    (703, 302, (664, 296, 742, 309), 2073000156),
    (790, 302, (752, 296, 829, 309), 2036844555),
    (879, 302, (839, 296, 919, 309), 1741666418),
    (967, 302, (930, 296, 1004, 309), 1696843700),
]
KPI_ALT = [2215384906, 2176718253, 1861204511, 1813297064]

BASE = 448
QBARS = [  # x0, x1, top, valor
    (733, 775, 364, 1002373632),
    (826, 868, 361, 1047108905),
    (920, 961, 445, 23457419),
]
Q_ALT = [862041324, 1172961974, 180381608]
PX_PER_EUR = (BASE - 364) / 1002373632

DIST_VALS = [207951837, 91337905, 90707997, 77354224, 65007136, 56636973, 53069642, 48986237, 47257530,
             42614022, 36853265, 33897161, 32386306, 31652460, 28224441, 25240946, 24336258, 23145801]
PROD_VALS = [249822888, 205205538, 172543529, 166383997, 129009242, 125133136, 121659778, 101473593, 98840373,
             67723837, 52716368, 35885296, 32834060, 32484165, 25197338, 23542073, 21462395, 17825340]


def teal_rows(x0, vals, label_end):
    f = orig.astype(int)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    teal = (g - r > 30) & (b - r > 20)
    rows = []
    for i, v in enumerate(vals):
        y0 = 495 + 10 * i
        xs = np.where(teal[y0:y0 + 8, x0 - 4:x0 + 80].any(0))[0]
        if not len(xs):
            continue
        xs_, xe = x0 - 4 + xs.min(), x0 - 4 + xs.max() + 1
        rows.append(dict(y0=y0 - 1, y1=y0 + 9, xs=xs_, xe=xe, val=v, xr=label_end,
                         bg=col(label_end - 15, 487), color=col(xs_ + 3, y0 + 4)))
    return rows


DIST = teal_rows(725, DIST_VALS, 826)
PROD = teal_rows(903, PROD_VALS, 997)
rng = np.random.default_rng(11)
for rows in (DIST, PROD):
    f = np.sort(rng.uniform(0.7, 1.25, len(rows)))[::-1]
    alt = sorted([r["val"] * k for r, k in zip(rows, f)], reverse=True)
    alt[0] = min(alt[0], rows[0]["val"] * 1.0)
    for r, a in zip(rows, alt):
        r["alt"] = min(a, alt[0])

DONUTS = [(1103, 357, 27, 13), (1328, 357, 27, 13)]

TREE = (1022, 486, 1458, 662)  # panel del árbol
TREE_BG = col(1440, 600)

# barras oscuras del árbol (relleno dentro de la pista gris)
_t = orig[TREE[1]:TREE[3], TREE[0]:TREE[2]].astype(int)
_dark = (_t.max(2) < 120) & (_t.max(2) - _t.min(2) < 25)
_n, _lab, _st, _ = cv2.connectedComponentsWithStats(_dark.astype(np.uint8))
TREE_BARS = []
for i in range(1, _n):
    x, y, w, h, a = _st[i]
    tr = _t[min(y + h // 2, _t.shape[0] - 1), min(x + w + 2, _t.shape[1] - 1)]
    solid = a >= 0.85 * w * h                       # rectángulo macizo, no un glifo
    on_track = 170 <= tr.max() <= 230 and tr.max() - tr.min() < 25
    if 2 <= h <= 6 and w >= 3 and a >= 8 and solid and on_track:
        TREE_BARS.append(dict(x=x + TREE[0], y=y + TREE[1], w=w, h=h,
                              track=col(min(x + TREE[0] + w + 3, 1456), y + TREE[1] + h // 2),
                              alt=float(rng.uniform(0.6, 1.35))))

# ---------------------------------------------------------------- guion


def state(t):
    """escala de construcción (0..1) y mezcla hacia datos alternativos (0..1)"""
    build = ease(t / 2.4) if t < 9.0 else 1 - ease_io((t - 9.0) / 0.9)
    if t < 4.2:
        alt = 0.0
    elif t < 5.6:
        alt = ease_io((t - 4.2) / 1.4)
    elif t < 7.0:
        alt = 1.0
    elif t < 8.4:
        alt = 1 - ease_io((t - 7.0) / 1.4)
    else:
        alt = 0.0
    return build, alt


def stagger(build, i, n, spread=0.45):
    """escalonado: cada elemento empieza un poco después que el anterior"""
    if build >= 1:
        return 1.0
    start = spread * i / max(n - 1, 1)
    return ease_io((build - start) / (1 - spread)) if build > start else 0.0


# fondo del gráfico de barras sin barras (conserva la rejilla)
BAR_BG = big0[:, (786 - RX0) * S:(818 - RX0) * S].copy()


def render(t):
    build, alt = state(t)
    if build >= 1 and alt == 0:
        return None  # estado original: sin cambios
    img = big0.copy()

    # --- barras trimestrales
    x0, y0 = P(728, 345)
    x1, y1 = P(968, BASE)
    w = x1 - x0
    tile = np.tile(BAR_BG, (1, w // BAR_BG.shape[1] + 2, 1))[:, :w]
    img[y0:y1, x0:x1] = tile[y0:y1]
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for i, (bx0, bx1, top, v) in enumerate(QBARS):
        val = (v + (Q_ALT[i] - v) * alt) * stagger(build, i, 3, 0.3)
        h = max(val * PX_PER_EUR, 0)
        if h > 0.2:
            d.rectangle([*P(bx0, BASE - h), P(bx1, BASE)[0] - 1, P(0, BASE)[1] - 1], fill=BLUE)
        if val > 0:
            s = euro(val)
            tw = d.textlength(s, font=F_BAR)
            cx = P((bx0 + bx1) / 2, 0)[0]
            d.text((cx - tw / 2, P(0, BASE - h - 8)[1]), s, fill=TEXT, font=F_BAR)

    # --- KPIs
    for i, (cx, cy, box, v) in enumerate(KPI):
        d.rectangle([*P(box[0], box[1]), *P(box[2], box[3])], fill=col(box[0] - 2, cy))
        val = (v + (KPI_ALT[i] - v) * alt) * ease(build)
        s = euro(val)
        tw = d.textlength(s, font=F_KPI)
        d.text((P(cx, 0)[0] - tw / 2, P(0, cy - 5.5)[1]), s, fill=TEXT, font=F_KPI)

    # --- rankings horizontales
    for rows in (DIST, PROD):
        vmax = rows[0]["val"]
        full = rows[0]["xe"] - rows[0]["xs"]
        for i, r in enumerate(rows):
            val = (r["val"] + (r["alt"] - r["val"]) * alt) * stagger(build, i, len(rows))
            L = full * val / vmax
            d.rectangle([*P(r["xs"], r["y0"]), *P(r["xr"], r["y1"])], fill=r["bg"])
            if L > 0.3:
                d.rectangle([*P(r["xs"], r["y0"] + 1), P(r["xs"] + L, 0)[0], P(0, r["y1"] - 1)[1] - 1],
                            fill=r["color"])
            if val > 0:
                d.text((P(r["xs"] + L + 4, 0)[0], P(0, r["y0"] + 1.3)[1]), euro(val), fill=TEXT, font=F_LAB)
    img = np.array(pil)

    # --- árbol: barras oscuras + aparición por columnas
    for b in TREE_BARS:
        nw = b["w"] * (1 + (b["alt"] - 1) * alt)
        xa, ya = P(b["x"], b["y"])
        _, yb = P(0, b["y"] + b["h"])
        xb = P(min(b["x"] + max(b["w"], nw), 1456), 0)[0]
        img[ya:yb, xa:xb] = b["track"]
        img[ya:yb, xa:min(xa + int(nw * S), xb)] = big0[ya:yb, xa + S:xa + S + 1]
    if build < 1:
        tx0, ty0 = P(TREE[0], TREE[1])
        tx1, ty1 = P(TREE[2], TREE[3])
        xs = np.arange(tx0, tx1)
        # frente de aparición de izquierda a derecha, suave
        front = tx0 + (tx1 - tx0 + 80 * S) * build - 40 * S
        a = np.clip((front - xs) / (40 * S), 0, 1)[None, :, None]
        img[ty0:ty1, tx0:tx1] = (img[ty0:ty1, tx0:tx1] * a + np.array(TREE_BG) * (1 - a)).astype(np.uint8)

    # --- donuts
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for i, (cx, cy, ro, ri) in enumerate(DONUTS):
        k = stagger(build, i, 2, 0.2)
        if k >= 1:
            continue
        (X, Y), R, Ri = P(cx, cy), ro * S, ri * S
        ya, yb, xa, xb = Y - R - 3 * S, Y + R + 3 * S, X - R - 3 * S, X + R + 3 * S
        dy, dx = yy[ya:yb, xa:xb] - Y, xx[ya:yb, xa:xb] - X
        rr = np.hypot(dx, dy)
        ang = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360
        hide = (rr <= R + S) & (rr >= Ri - S) & (ang >= 360 * k)
        img[ya:yb, xa:xb][hide] = col(cx, cy)
    return img


def composite(frame, img):
    if img is None:
        return
    small = cv2.resize(img, (RX1 - RX0, RY1 - RY0), interpolation=cv2.INTER_AREA)
    diff = np.abs(small.astype(np.int16) - small0.astype(np.int16)).max(2) > 3
    m = cv2.GaussianBlur(cv2.dilate(diff.astype(np.uint8) * 255, np.ones((3, 3), np.uint8)), (3, 3), 0)
    m = m.astype(np.float32)[..., None] / 255
    ya, xa = RY0 + DY, RX0 + DX
    reg = frame[ya:ya + RY1 - RY0, xa:xa + RX1 - RX0].astype(np.float32)
    frame[ya:ya + RY1 - RY0, xa:xa + RX1 - RX0] = (reg * (1 - m) + small * m).astype(np.uint8)


def main():
    out = os.path.join(HERE, PROF["out"] + ".mp4")
    w = imageio_ffmpeg.write_frames(out, (IW, IH - IH % 2), fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                                    output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    w.send(None)
    for i in range(NF):
        frame = photo.copy()
        composite(frame, render(i / FPS))
        w.send(np.ascontiguousarray(frame[:IH - IH % 2]))
    w.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.system(f'"{ff}" -y -loglevel error -i "{out}" -vf "fps=15,scale=900:-1:flags=lanczos,split[a][b];'
              f'[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:'
              f'diff_mode=rectangle" "{os.path.join(HERE, PROF["out"] + ".gif")}"')


if __name__ == "__main__":
    main()
