"""Zoom de cámara sobre la pantalla de GA Services con un cursor que navega.

La cámara parte de la vista completa y se acerca poco a poco siguiendo al
cursor, que pasa por varias tarjetas (efecto hover) y termina haciendo clic
en GARAGES.

Uso: python3 animar_navegacion.py  ->  navegacion.mp4 (1920x1080) y navegacion.gif
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "navegacion_original.webp")
OW, OH = 1920, 1080
FPS, DUR = 30, 11.0
NF = int(FPS * DUR)

base = np.array(Image.open(SRC).convert("RGB"))
SH, SW = base.shape[:2]
BG = base[800, 1700].astype(np.float32)

# borra el cursor que venía en la captura
base[834:860, 1838:1860] = base[800, 1700]

# tarjetas: (x0, y0, x1, y1) en coordenadas de la captura
XS = [(492, 631), (858, 995), (1224, 1361), (1590, 1727)]
YS = [(150, 286), (312, 448), (473, 610), (633, 771)]
TILES = {}
names = [["GARAGES", "SUBSCRIPTIONS", "AUDITS", "PROMOTIONS"],
         ["TRAINING", "CALENDAR", "ALERTS", "WEB"],
         ["PERMISSIONS", "ROLES", "REPAIR", "SMS"],
         ["STATISTIC", None, None, None]]
for r, (y0, y1) in enumerate(YS):
    for c, (x0, x1) in enumerate(XS):
        if names[r][c]:
            TILES[names[r][c]] = (x0, y0, x1, y1)


def center(n):
    x0, y0, x1, y1 = TILES[n]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2 - 12])


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- guion
START = np.array([1848.0, 846.0])
PATH = [  # (t, punto)
    (0.0, START), (0.7, START),
    (1.9, center("PROMOTIONS") + [16, -6]), (2.7, center("PROMOTIONS") + [16, -6]),
    (3.7, center("ALERTS") + [12, -4]), (4.5, center("ALERTS") + [12, -4]),
    (5.5, center("CALENDAR") + [14, -2]), (6.2, center("CALENDAR") + [14, -2]),
    (7.3, center("GARAGES") + [10, -4]), (DUR, center("GARAGES") + [10, -4]),
]
CLICK = 7.75

# cámara: (t, zoom, centro) — el centro se suaviza siguiendo al cursor
CAM = [
    (0.0, 1.0, np.array([SW / 2, SH / 2])),
    (1.0, 1.0, np.array([SW / 2, SH / 2])),
    (4.0, 1.45, np.array([1300.0, 400.0])),
    (7.0, 1.8, np.array([820.0, 350.0])),
    (9.2, 2.15, np.array([700.0, 300.0])),
    (DUR, 2.25, np.array([690.0, 295.0])),
]


def interp(keys, t):
    for (ta, *a), (tb, *b) in zip(keys, keys[1:]):
        if ta <= t <= tb:
            k = ease_io((t - ta) / (tb - ta)) if tb > ta else 1.0
            return [ai + (bi - ai) * k for ai, bi in zip(a, b)]
    return list(keys[-1][1:])


def cursor_pos(t):
    for (ta, pa), (tb, pb) in zip(PATH, PATH[1:]):
        if ta <= t <= tb:
            k = ease_io((t - ta) / (tb - ta))
            # ligera curva en los desplazamientos, como una mano real
            d = pb - pa
            bow = np.array([-d[1], d[0]]) * 0.08 * np.sin(np.pi * k)
            return pa + d * k + bow
    return PATH[-1][1]


def camera(t):
    z, c = interp(CAM, t)
    # la ventana visible nunca sale de la captura
    vw, vh = SW / z, SH / z
    cx = min(max(c[0], vw / 2), SW - vw / 2)
    cy = min(max(c[1], vh / 2), SH - vh / 2)
    return z, np.array([cx, cy])


def hover_amount(name, t):
    x0, y0, x1, y1 = TILES[name]
    # suavizado temporal: se calcula sobre una ventana corta
    acc = 0.0
    for dt in np.linspace(-0.18, 0, 5):
        p = cursor_pos(max(t + dt, 0))
        acc += 1.0 if (x0 - 4 <= p[0] <= x1 + 4 and y0 - 4 <= p[1] <= y1 + 4) else 0.0
    return acc / 5


def render_screen(t):
    img = base.astype(np.float32).copy()
    for name, (x0, y0, x1, y1) in TILES.items():
        h = hover_amount(name, t)
        press = max(0.0, 1 - abs(t - CLICK) / 0.12) if name == "GARAGES" else 0.0
        if h <= 0 and press <= 0:
            continue
        M = 22
        X0, Y0, X1, Y1 = x0 - M, y0 - M, x1 + M, y1 + M
        patch = base[Y0:Y1, X0:X1].copy()
        scale = 1 + 0.045 * h - 0.03 * press
        lift = -4 * h + 2 * press
        # sombra difusa bajo la tarjeta
        P = 60  # margen amplio para que el desenfoque llegue a cero antes del borde
        sh = np.zeros((Y1 - Y0 + 2 * P, X1 - X0 + 2 * P), np.float32)
        cv2.rectangle(sh, (M + P + 4, int(M + P + 10 + lift)), (x1 - X0 + P - 4, int(y1 - Y0 + P + 14 + lift)), 1.0, -1)
        sh = np.clip((cv2.GaussianBlur(sh, (0, 0), 12) - 0.03) / 0.97, 0, 1)
        reg = img[Y0 - P:Y1 + P, X0 - P:X1 + P]
        reg *= (1 - 0.16 * h * sh)[..., None]
        # tarjeta escalada y elevada
        pw, ph = X1 - X0, Y1 - Y0
        A = cv2.getRotationMatrix2D((pw / 2, ph / 2), 0, scale)
        A[1, 2] += lift
        warped = cv2.warpAffine(patch, A, (pw, ph), flags=cv2.INTER_CUBIC, borderValue=BG.tolist())
        m0 = np.zeros((ph, pw), np.float32)
        m0[M:M + y1 - y0, M:M + x1 - x0] = 1
        badge = base[y0 + 3, x0 + 3].astype(int)
        if badge[0] - badge[2] > 40 or badge[0] > 200 and badge[1] < 245:  # insignia roja con número
            cv2.circle(m0, (M + 3, M + 3), 16, 1.0, -1, cv2.LINE_AA)
        m = cv2.warpAffine(m0, A, (pw, ph), flags=cv2.INTER_LINEAR)[..., None]
        # borde azul inferior al pasar el ratón
        if h > 0:
            wb = warped.astype(np.float32)
            by0 = int(ph / 2 + (y1 - y0) / 2 * scale + lift) - 3
            bx0, bx1 = int(pw / 2 - (x1 - x0) / 2 * scale), int(pw / 2 + (x1 - x0) / 2 * scale)
            wb[by0:by0 + 3, bx0:bx1] = wb[by0:by0 + 3, bx0:bx1] * (1 - h) + np.array([42, 90, 160]) * h
            warped = wb
        img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - m) + warped.astype(np.float32) * m
    # onda del clic
    if CLICK <= t < CLICK + 0.7:
        k = (t - CLICK) / 0.7
        p = cursor_pos(t)
        ov = np.zeros((SH, SW), np.uint8)
        cv2.circle(ov, (int(p[0] * 4), int(p[1] * 4)), int((8 + 60 * k) * 4), 255, -1, cv2.LINE_AA, shift=2)
        a = cv2.GaussianBlur(ov, (5, 5), 0).astype(np.float32)[..., None] / 255 * 0.28 * (1 - k)
        img = img * (1 - a) + np.array([42, 90, 160], np.float32) * a
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_cursor(frame, p, size, press):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil, "RGBA")
    s = size * (1 - 0.12 * press)
    pts = [(0, 0), (0, 17), (4.2, 13), (7, 19.5), (9.6, 18.4), (6.9, 12.2), (12.2, 12.2)]
    x, y = p
    d.polygon([(x + px * s + s * 0.9, y + py * s + s * 1.2) for px, py in pts], fill=(0, 0, 0, 55))
    d.polygon([(x + px * s, y + py * s) for px, py in pts], fill=(255, 255, 255, 255))
    d.line([(x + px * s, y + py * s) for px, py in pts + [pts[0]]], fill=(20, 20, 20, 255), width=max(1, int(s)))
    return np.array(pil)


def render(t):
    scr = render_screen(t)
    z, c = camera(t)
    k = OW / SW * z  # escala captura -> salida
    A = np.float32([[k, 0, OW / 2 - c[0] * k], [0, k, OH / 2 - c[1] * k]])
    out = cv2.warpAffine(scr, A, (OW, OH), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    if z > 1.05:  # enfoque suave para compensar la ampliación
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        amt = min(0.6, (z - 1) * 0.5)
        out = cv2.addWeighted(out, 1 + amt, blur, -amt, 0)
    p = cursor_pos(t)
    cp = A @ np.array([p[0], p[1], 1.0])
    press = max(0.0, 1 - abs(t - CLICK) / 0.12)
    # el cursor crece algo con el zoom, pero menos que la pantalla
    return draw_cursor(out, cp, 1.05 * k ** 0.6, press)


def main():
    out = os.path.join(HERE, "navegacion.mp4")
    w = imageio_ffmpeg.write_frames(out, (OW, OH), fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                                    output_params=["-crf", "17", "-preset", "slow", "-movflags", "+faststart"])
    w.send(None)
    for i in range(NF):
        w.send(np.ascontiguousarray(render(i / FPS)))
    w.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.system(f'"{ff}" -y -loglevel error -i "{out}" -vf "fps=15,scale=900:-1:flags=lanczos,split[a][b];'
              f'[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:'
              f'diff_mode=rectangle" "{os.path.join(HERE, "navegacion.gif")}"')


if __name__ == "__main__":
    main()
