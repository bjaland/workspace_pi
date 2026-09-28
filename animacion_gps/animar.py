"""Anima la foto del portátil con el dashboard GPS.

- Mano izquierda tecleando (dedos que se levantan y golpean).
- Mano derecha deslizando y haciendo clic en el trackpad.
- En pantalla: cursor, escritura en el filtro "Supplier / Brand",
  lista de sugerencias, refresco de barras, donuts y KPIs, tooltip.

Uso: python3 animar.py  ->  genera gps_animacion.mp4 y gps_animacion.gif
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "original.jpg")
FPS = 30
DUR = 10.0
NF = int(FPS * DUR)

orig = cv2.imread(SRC)[:, :, ::-1].copy()  # RGB
IH, IW = orig.shape[:2]

# ---------------------------------------------------------------- pantalla
SCR = np.float32([[783, 95], [1595, 400], [1455, 940], [700, 620]])
FW, FH = 1600, 1040
M = cv2.getPerspectiveTransform(SCR, np.float32([[0, 0], [FW, 0], [FW, FH], [0, FH]]))
flat0 = cv2.warpPerspective(orig, M, (FW, FH), flags=cv2.INTER_CUBIC)
BX0, BY0, BX1, BY1 = 690, 90, 1600, 950  # bbox de la pantalla en la foto
T = np.array([[1, 0, -BX0], [0, 1, -BY0], [0, 0, 1]], np.float64)
Minv = T @ np.linalg.inv(M)

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONTB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
f_kpi = ImageFont.truetype(FONT, 14)
f_field = ImageFont.truetype(FONT, 11)
f_fieldb = ImageFont.truetype(FONTB, 11)
f_small = ImageFont.truetype(FONT, 8)
f_tip = ImageFont.truetype(FONT, 11)
f_tipb = ImageFont.truetype(FONTB, 11)

BLUE = (19, 89, 158)
TEXT = (90, 90, 95)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def euro(v):
    return "€ " + f"{int(round(v)):,}"


# --- estados de datos: "All" (original) y "BOSCH" (filtrado)
KPI_POS = [(383, 196), (503, 196), (620, 196), (744, 196)]
KPI_BOX = [(334, 187, 434, 205), (452, 187, 556, 205), (570, 187, 672, 205), (692, 187, 796, 205)]
KPI_ALL = [2073000156, 2036844555, 1741666418, 1696843700]
KPI_BOSCH = [990466231, 972905318, 831774062, 810337915]

BARS = [dict(x0=424, x1=482, top=282), dict(x0=551, x1=609, top=278)]
BASE = 396
Q_ALL = [1002573602, 1047198935]
Q_BOSCH = [480112540, 498902117]

rng = np.random.default_rng(7)


def teal_rows(x0, x1, y_start, pitch, n, xr):
    rows = []
    b, g, r = flat0[..., 2].astype(int), flat0[..., 1].astype(int), flat0[..., 0].astype(int)
    teal = (g - r > 40) & (b - r > 25)
    for i in range(n):
        ya = int(round(y_start + i * pitch))
        yb = ya + 10
        xs = np.where(teal[ya:yb, x0:x1].any(0))[0]
        if len(xs) == 0:
            continue
        bg = np.median(flat0[ya - 1:yb + 1, xr - 4:xr].reshape(-1, 3), axis=0).astype(np.uint8)
        rows.append(dict(y0=ya - 1, y1=yb + 1, xs=x0 + xs.min(), xe=x0 + xs.max() + 1, xr=xr, bg=bg))
    return rows


TEAL = teal_rows(405, 560, 463, 13.75, 18, 552) + teal_rows(650, 800, 464, 13.9, 17, 790)
for i, rw in enumerate(TEAL):
    rw["bosch"] = 1.0 if i in (0, 18) else float(rng.uniform(0.3, 0.85))

DONUTS = [(929, 276, 38, 23), (1236, 277, 38, 23)]
FIELD = (845, 138, 1012, 158)
ERASER = (982, 148)

# ---------------------------------------------------------------- guion
REST = np.array([700.0, 640.0])
P_FIELD = np.array([915.0, 150.0])
P_SUG = np.array([905.0, 172.0])
P_AWAY = np.array([1040.0, 108.0])          # aparta el cursor mientras escribe
P_BAR = np.array([585.0, 335.0])
P_ERASE = np.array([ERASER[0] + 2.0, ERASER[1] + 2.0])

TYPED = "Bosch Aftermarket"
_trng = np.random.default_rng(3)
KEY_T = []
_t = 1.45
for ch in TYPED:
    KEY_T.append(round(_t, 3))
    _t += float(_trng.uniform(0.11, 0.2)) + (0.12 if ch == " " else 0)
T_TYPED = KEY_T[-1]
KEY_FINGER = [2 if ch == " " else int(_trng.choice([0, 1, 3])) for ch in TYPED]
T_HOVER = T_TYPED + 0.7
T_SEL = T_TYPED + 0.85                      # clic en la sugerencia
T_TIP0, T_TIP1 = T_SEL + 3.1, T_SEL + 4.5  # tooltip
T_ERASE = T_TIP1 + 0.75                    # clic en borrar filtro
CLICKS = [1.12, T_SEL, T_ERASE]
REFRESH = [(T_SEL + 0.1, "BOSCH"), (T_ERASE + 0.1, "ALL")]
REFRESH_DUR = 1.5
BRANDS = ["BOSCH AFTERMARKET", "BOSCH CAR SERVICE", "BOSAL", "BORGWARNER", "BREMBO", "BLUE PRINT"]
SELECTED = BRANDS[0]
DUR = float(np.ceil((T_ERASE + 2.4) * 2) / 2)
NF = int(FPS * DUR)

CURSOR_KEYS = [  # (t, pos)
    (0.0, REST), (0.25, REST), (1.05, P_FIELD), (1.2, P_FIELD), (1.7, P_AWAY), (T_TYPED + 0.15, P_AWAY), (T_HOVER, P_SUG),
    (T_SEL + 2.2, P_SUG), (T_TIP0 - 0.1, P_BAR), (T_TIP1, P_BAR), (T_ERASE - 0.1, P_ERASE),
    (T_ERASE + 0.4, P_ERASE), (DUR - 0.35, REST), (DUR, REST),
]


def cursor_pos(t):
    for (ta, pa), (tb, pb) in zip(CURSOR_KEYS, CURSOR_KEYS[1:]):
        if ta <= t <= tb:
            k = ease_io((t - ta) / (tb - ta)) if tb > ta else 1
            # pequeña curva para que no sea una recta perfecta
            mid = (pa + pb) / 2 + np.array([0, -18]) * np.sin(np.pi * k) * (np.linalg.norm(pb - pa) > 50)
            return (1 - k) ** 2 * pa + 2 * (1 - k) * k * mid + k * k * pb
    return REST


def click_amount(t):
    return max((1 - abs(t - c) / 0.09 for c in CLICKS), default=0) if CLICKS else 0


def data_state(t):
    """Devuelve (valores_desde, valores_hacia, progreso) para el refresco."""
    cur, prev, p = "ALL", "ALL", 1.0
    for tr, st in REFRESH:
        if t >= tr:
            prev, cur = cur, st
            p = (t - tr) / REFRESH_DUR
    return prev, cur, p


def values(state):
    if state == "ALL":
        return KPI_ALL, Q_ALL, [1.0] * len(TEAL)
    return KPI_BOSCH, Q_BOSCH, [r["bosch"] for r in TEAL]


# fondo del gráfico de barras (tira sin barras, conserva líneas de rejilla)
BAR_BG = flat0[250:BASE + 1, 492:540].copy()


def fill_from_tile(img, x0, y0, x1, y1, tile):
    w = x1 - x0
    reps = int(np.ceil(w / tile.shape[1])) + 1
    strip = np.tile(tile, (1, reps, 1))[:, :w]
    img[y0:y1, x0:x1] = strip[(y0 - 250):(y1 - 250)]


def draw_cursor(d, p, press):
    x, y = p
    s = 1.0 - 0.12 * press
    pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    pts = [(x + px * s, y + py * s) for px, py in pts]
    sh = [(px + 1.5, py + 1.5) for px, py in pts]
    d.polygon(sh, fill=(150, 150, 150))
    d.polygon(pts, fill=(255, 255, 255), outline=(0, 0, 0))


def render_screen(t):
    img = flat0.copy()
    prev, cur, p = data_state(t)
    k = ease(p)
    kpi_a, q_a, teal_a = values(prev)
    kpi_b, q_b, teal_b = values(cur)
    anim = p < 1.0

    if anim or cur != "ALL":
        # barras trimestrales
        fill_from_tile(img, 418, 258, 616, BASE, BAR_BG)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for bar, va, vb, qo in zip(BARS, q_a, q_b, Q_ALL):
            v = lerp(va, vb, k)
            full_h = BASE - bar["top"]
            h = full_h * v / qo
            top = BASE - h
            d.rectangle([bar["x0"], top, bar["x1"] - 1, BASE - 1], fill=BLUE)
            lbl = euro(v)
            tw = d.textlength(lbl, font=f_small)
            d.text(((bar["x0"] + bar["x1"]) / 2 - tw / 2, top - 11), lbl, fill=TEXT, font=f_small)
        img = np.array(pil)

        # barras horizontales
        for rw, fa, fb in zip(TEAL, teal_a, teal_b):
            fr = lerp(fa, fb, k)
            L = rw["xe"] - rw["xs"]
            nl = max(1, int(round(L * fr)))
            seg = flat0[rw["y0"]:rw["y1"], rw["xs"]:rw["xe"]]
            lw = rw["xr"] - rw["xe"]
            lab = flat0[rw["y0"]:rw["y1"], rw["xe"]:rw["xr"]]
            img[rw["y0"]:rw["y1"], rw["xs"]:rw["xr"]] = rw["bg"]
            img[rw["y0"]:rw["y1"], rw["xs"]:rw["xs"] + nl] = cv2.resize(seg, (nl, seg.shape[0]), interpolation=cv2.INTER_AREA)
            img[rw["y0"]:rw["y1"], rw["xs"] + nl:rw["xs"] + nl + lw] = lab

        # KPIs
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for (cx, cy), box, va, vb in zip(KPI_POS, KPI_BOX, kpi_a, kpi_b):
            bg = tuple(int(c) for c in flat0[box[1] + 2, box[0] + 3])
            d.rectangle(box, fill=bg)
            s = euro(lerp(va, vb, k))
            tw = d.textlength(s, font=f_kpi)
            d.text((cx - tw / 2, cy - 8), s, fill=TEXT, font=f_kpi)
        img = np.array(pil)

    # donuts: barrido al refrescar
    if anim:
        yy, xx = np.mgrid[0:FH, 0:FW]
        for cx, cy, ro, ri in DONUTS:
            y0, y1, x0, x1 = cy - ro - 2, cy + ro + 3, cx - ro - 2, cx + ro + 3
            dy, dx = yy[y0:y1, x0:x1] - cy, xx[y0:y1, x0:x1] - cx
            rr = np.hypot(dx, dy)
            ang = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360
            ring = (rr <= ro + 1) & (rr >= ri - 1)
            hide = ring & (ang > 360 * ease_io(p / 0.75))
            reg = img[y0:y1, x0:x1]
            reg[hide] = (252, 252, 253)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)

    # campo Supplier / Brand
    fx0, fy0, fx1, fy1 = FIELD
    focused = CLICKS[0] <= t < CLICKS[1]
    typed_n = sum(1 for kt in KEY_T if t >= kt) if focused else 0
    selected = CLICKS[1] <= t < REFRESH[1][0]
    if focused or selected:
        d.rectangle([fx0 + 10, fy0 + 3, fx1 - 20, fy1 - 2], fill=(250, 250, 252))
        if focused:
            d.rectangle([fx0 + 7, fy0, fx1 - 5, fy1 + 1], outline=(40, 110, 200), width=1)
            txt = TYPED[:typed_n]
            d.text((fx0 + 14, fy0 + 5), txt, fill=(40, 40, 45), font=f_field)
            if int(t * 2.2) % 2 == 0 or (typed_n and t - KEY_T[typed_n - 1] < 0.3):
                cx = fx0 + 14 + d.textlength(txt, font=f_field) + 1
                d.line([(cx, fy0 + 4), (cx, fy0 + 16)], fill=(30, 30, 30), width=1)
            q = txt.upper()
            sug = [b for b in BRANDS if q and b.startswith(q)][:3]
            if sug:
                hover = t > T_HOVER - 0.1
                lx0, ly0 = fx0 + 7, fy1 + 3
                lh = 16
                d.rectangle([lx0 + 2, ly0 + 2, fx1 - 3, ly0 + lh * len(sug) + 6], fill=(205, 205, 210))
                d.rectangle([lx0, ly0, fx1 - 5, ly0 + lh * len(sug) + 4], fill=(255, 255, 255), outline=(200, 200, 205))
                for i, b in enumerate(sug):
                    yy0 = ly0 + 2 + i * lh
                    if i == 0 and hover:
                        d.rectangle([lx0 + 1, yy0, fx1 - 6, yy0 + lh - 1], fill=(222, 234, 250))
                    d.rectangle([lx0 + 6, yy0 + 4, lx0 + 13, yy0 + 11], outline=(120, 120, 125),
                                fill=(40, 110, 200) if (i == 0 and t > CLICKS[1] - 0.02) else None)
                    d.text((lx0 + 19, yy0 + 3), b, fill=(40, 40, 45),
                           font=f_fieldb if len(q) and i == 0 and hover else f_field)
        else:
            d.text((fx0 + 14, fy0 + 5), SELECTED, fill=(40, 40, 45), font=f_field)
            ex, ey = ERASER
            d.line([(ex - 3, ey - 3), (ex + 3, ey + 3)], fill=(90, 90, 95), width=1)
            d.line([(ex - 3, ey + 3), (ex + 3, ey - 3)], fill=(90, 90, 95), width=1)

    # tooltip sobre la barra Q2
    if T_TIP0 <= t < T_TIP1:
        a = min(1, (t - T_TIP0) / 0.15, (T_TIP1 - t) / 0.15)
        cp = cursor_pos(t)
        x0, y0 = cp[0] + 16, cp[1] + 14
        _, qv, _ = values(cur)
        lines = [("Quarter", "Q2 2026"), ("Supplier", "BOSCH"), ("Purchases", euro(qv[1]))]
        over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        od.rectangle([x0 + 3, y0 + 3, x0 + 185, y0 + 61], fill=(0, 0, 0, int(60 * a)))
        od.rectangle([x0, y0, x0 + 182, y0 + 58], fill=(45, 48, 55, int(245 * a)))
        for i, (kk, vv) in enumerate(lines):
            od.text((x0 + 9, y0 + 6 + i * 16), kk, fill=(190, 195, 205, int(255 * a)), font=f_tip)
            od.text((x0 + 78, y0 + 6 + i * 16), vv, fill=(255, 255, 255, int(255 * a)), font=f_tipb)
        pil = Image.alpha_composite(pil.convert("RGBA"), over).convert("RGB")
        d = ImageDraw.Draw(pil)

    draw_cursor(d, cursor_pos(t), click_amount(t))
    return np.array(pil)


def composite_screen(frame, flat):
    diff = np.abs(flat.astype(np.int16) - flat0.astype(np.int16)).max(2) > 2
    if not diff.any():
        return
    mask = cv2.dilate(diff.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    w, h = BX1 - BX0, BY1 - BY0
    wf = cv2.warpPerspective(flat, Minv, (w, h), flags=cv2.INTER_AREA)
    wm = cv2.warpPerspective(mask, Minv, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    reg = frame[BY0:BY1, BX0:BX1].astype(np.float32)
    frame[BY0:BY1, BX0:BX1] = (reg * (1 - wm) + wf * wm).astype(np.uint8)


# ---------------------------------------------------------------- manos
HX0, HY0, HX1, HY1 = 0, 640, 1260, 1500
hand = orig[HY0:HY1, HX0:HX1].copy()
hsv = cv2.cvtColor(hand, cv2.COLOR_RGB2HSV)
hh, ss, vv = cv2.split(hsv)
skin = (((hh < 25) | (hh > 170)) & (ss > 40) & (vv > 70)).astype(np.uint8) * 255
skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
hole = cv2.dilate(skin, np.ones((9, 9), np.uint8))
inp = cv2.inpaint(hand[:, :, ::-1], hole, 7, cv2.INPAINT_TELEA)[:, :, ::-1]
# fondo: la foto original salvo donde estaba la mano (ahí, relleno)
sw = cv2.GaussianBlur(cv2.dilate(skin, np.ones((5, 5), np.uint8)), (5, 5), 0).astype(np.float32)[..., None] / 255
bg_plate = (hand * (1 - sw) + inp * sw).astype(np.uint8)
skin_f = cv2.GaussianBlur(skin, (3, 3), 0).astype(np.float32) / 255

gy, gx = np.mgrid[HY0:HY1, HX0:HX1].astype(np.float32)


def unit(v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


# dedos de la mano izquierda: punta, eje (nudillo -> punta), longitud
FINGERS = [
    dict(tip=(776, 824), axis=unit((0.95, 0.3)), L=150, sig=14),
    dict(tip=(810, 880), axis=unit((0.97, 0.24)), L=160, sig=15),
    dict(tip=(737, 940), axis=unit((1.0, 0.07)), L=120, sig=16),   # pulgar
    dict(tip=(690, 778), axis=unit((0.9, 0.42)), L=120, sig=13),
]
RIGHT_INDEX = dict(tip=(806, 1127), axis=unit((0.25, -0.97)), L=110, sig=15)


def finger_weight(f):
    px, py = f["tip"]
    ax, ay = f["axis"]
    rx, ry = gx - px, gy - py
    s = rx * ax + ry * ay            # >0 más allá de la punta
    perp = -rx * ay + ry * ax
    along = np.where(s < 0, np.clip(1 + s / f["L"], 0, 1) ** 1.6, np.exp(-(s ** 2) / (2 * 18 ** 2)))
    return along * np.exp(-(perp ** 2) / (2 * f["sig"] ** 2))


FW_MAPS = [finger_weight(f) for f in FINGERS]
RW_MAP = finger_weight(RIGHT_INDEX)

# peso de la mano derecha completa (el antebrazo se mueve menos)
right_region = ((gy > 1040) & (gx > 560)).astype(np.float32)
right_region *= np.clip(1 - (gy - 1260) / 400, 0.25, 1)
right_region = cv2.GaussianBlur(right_region, (41, 41), 0)

DECK_R = unit((0.9, 0.43))
DECK_F = unit((0.28, -0.96))


def stroke(t, t0, up=0.13, down=0.05):
    """0 en reposo (tecla pulsada), 1 = dedo levantado."""
    if t0 - up <= t < t0:
        return np.sin(np.pi / 2 * (t - (t0 - up)) / (up - down)) if t < t0 - down else (t0 - t) / down
    return 0.0


def hand_fields(t):
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    # tecleo
    amt = [0.0] * len(FINGERS)
    for kt, fi in zip(KEY_T, KEY_FINGER):
        amt[fi] = max(amt[fi], stroke(t, kt, up=0.14, down=0.045))
    for fi, a in enumerate(amt):
        if a:
            f = FINGERS[fi]
            lift = -0.35 * f["axis"] + np.array([0, -1.0])
            lift = unit(lift) * (8 if fi == 2 else 11) * a
            dx += FW_MAPS[fi] * lift[0]
            dy += FW_MAPS[fi] * lift[1]
    # movimiento de reposo de los dedos (respiración sutil)
    for i, w in enumerate(FW_MAPS):
        a = 0.5 + 0.5 * np.sin(2 * np.pi * (t / DUR * 3) + i * 1.7)
        dy += w * (-1.8 * a)
    # mano derecha: sigue al cursor
    off = (cursor_pos(t) - REST) * 0.045
    o = DECK_R * off[0] + DECK_F * (-off[1])
    dx += right_region * o[0]
    dy += right_region * o[1]
    # clic del índice derecho
    for c in CLICKS:
        a = stroke(t, c, up=0.14, down=0.05)
        if a:
            dx += RW_MAP * 1.5 * a
            dy += RW_MAP * (-8 * a)
    return dx, dy


def composite_hands(frame, t):
    dx, dy = hand_fields(t)
    mx = (gx - dx - HX0).astype(np.float32)
    my = (gy - dy - HY0).astype(np.float32)
    warped = cv2.remap(hand, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wm = cv2.remap(skin_f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[..., None]
    out = bg_plate.astype(np.float32) * (1 - wm) + warped.astype(np.float32) * wm
    # sombra suave de contacto: oscurece ligeramente teclas cuando el dedo golpea
    frame[HY0:HY1, HX0:HX1] = out.astype(np.uint8)


# ---------------------------------------------------------------- render
def main():
    out_mp4 = os.path.join(HERE, "gps_animacion.mp4")
    writer = imageio_ffmpeg.write_frames(
        out_mp4, (IW, IH), fps=FPS, codec="libx264", quality=None, macro_block_size=1,
        output_params=["-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)
    gif_frames = []
    for i in range(NF):
        t = i / FPS
        frame = orig.copy()
        composite_hands(frame, t)
        composite_screen(frame, render_screen(t))
        writer.send(np.ascontiguousarray(frame))
        if i % 2 == 0:
            gif_frames.append(Image.fromarray(cv2.resize(frame, (720, 540), interpolation=cv2.INTER_AREA)))
    writer.close()
    gif_frames[0].save(os.path.join(HERE, "gps_animacion.gif"), save_all=True,
                       append_images=gif_frames[1:], duration=int(2000 / FPS), loop=0, optimize=True)


if __name__ == "__main__":
    print("duración", DUR, "s; tecleo hasta", T_TYPED)
    main()
