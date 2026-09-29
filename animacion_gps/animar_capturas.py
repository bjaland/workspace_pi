"""Zoom de cámara + navegación simulada sobre capturas que no son 16:9.

La captura empieza completa sobre un fondo neutro y la cámara se va
acercando mientras se navega:
  - web   : ficha del vehículo (G-Connect) con cursor de ratón, hovers,
            clic en "Current Status" que redibuja los indicadores y tooltips.
  - movil : app EuroGarage dentro de un marco de teléfono, con toques.

  - operaciones / presupuesto / recambios : capturas 16:9 del flujo de
            presupuesto (lista -> nuevo presupuesto -> recambios y cesta).

Uso: python3 animar_capturas.py web|movil|operaciones|presupuesto|recambios
     ->  <nombre>.mp4 (1920x1080) y .gif
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH, FPS = 1920, 1080, 30
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONTB = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
ACCENT = np.array([42, 110, 200], np.float32)


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def interp(keys, t):
    """keys: [(t, v0, v1, ...)] con interpolación suave entre claves"""
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            k = ease_io((t - a[0]) / (b[0] - a[0])) if b[0] > a[0] else 1.0
            return np.array([x + (y - x) * k for x, y in zip(a[1:], b[1:])], np.float64)
    return np.array(keys[-1][1:], np.float64)


def rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def backdrop():
    y, x = np.mgrid[0:OH, 0:OW].astype(np.float32)
    g = 1 - 0.06 * (((x - OW / 2) / OW) ** 2 + ((y - OH / 2) / OH) ** 2) * 2.2
    top, bot = np.array([236, 239, 244]), np.array([222, 226, 234])
    img = top[None, None] * (1 - y[..., None] / OH) + bot[None, None] * (y[..., None] / OH)
    return (img * g[..., None]).astype(np.float32)


BACK = backdrop()


# ---------------------------------------------------------------- perfiles
class Web:
    name = "ficha_vehiculo"
    src = "captura_web.png"
    DUR = 12.5
    pointer = "arrow"

    def __init__(self):
        self.screen = np.array(Image.open(os.path.join(HERE, self.src)).convert("RGB"))
        h, w = self.screen.shape[:2]
        self.R = 16
        a = rounded_mask(w, h, self.R)
        self.dev = np.dstack([self.screen, a])
        self.off = (0, 0)
        cur = np.array
        self.links = {"mas": (1133, 436, 1252, 454), "remota": (1113, 469, 1252, 487)}
        self.tabs = {"viajes": (606, 524, 763, 558), "status": (462, 524, 605, 558), "alertas": (263, 524, 354, 558)}
        self.gauges = [  # centro, radio exterior, rect hover, texto tooltip
            ((1017, 777), 78, (930, 695, 1105, 855), ("Neumáticos delanteros", "Aceptable · 35 %")),
            ((1408, 777), 78, (1321, 695, 1496, 855), ("Neumáticos traseros", "Normal · 62 %")),
            ((1017, 1063), 78, (930, 982, 1105, 1142), ("Frenos delanteros", "Aceptable · 27 %")),
            ((1408, 1063), 78, (1321, 982, 1496, 1142), ("Frenos traseros", "Aceptable · 27 %")),
        ]
        self._prep_gauges()
        P = lambda r, dx=0, dy=0: cur([(r[0] + r[2]) / 2 + dx, (r[1] + r[3]) / 2 + dy], float)
        self.PATH = [
            (0.0, cur([1480.0, 1210.0])), (0.9, cur([1480.0, 1210.0])),
            (2.3, P(self.links["mas"], 10, 3)), (3.0, P(self.links["mas"], 10, 3)),
            (3.9, P(self.tabs["viajes"], 5, 4)), (4.5, P(self.tabs["viajes"], 5, 4)),
            (5.1, P(self.tabs["status"], 8, 4)), (5.6, P(self.tabs["status"], 8, 4)),
            (7.3, cur([1030.0, 790.0])), (8.6, cur([1030.0, 790.0])),
            (9.6, cur([1420.0, 790.0])), (10.7, cur([1420.0, 790.0])),
            (11.6, cur([1400.0, 1075.0])), (self.DUR, cur([1400.0, 1075.0])),
        ]
        self.CLICKS = [5.35]
        self.REDRAW = 5.45  # los indicadores se redibujan tras el clic
        # cámara: (t, escala salida/captura, centro x, centro y)
        fit = (OH - 110) / h
        self.CAM = [
            (0.0, fit, w / 2, h / 2), (1.0, fit, w / 2, h / 2),
            (3.2, 1.25, 980, 500), (5.4, 1.32, 900, 700),
            (7.0, 1.75, 1180, 830), (10.6, 1.9, 1240, 840),
            (self.DUR, 1.95, 1230, 900),
        ]

    def _prep_gauges(self):
        s = self.screen.astype(int)
        sat = s.max(2) - s.min(2)
        self.arcs = []
        for (cx, cy), r, _, _ in self.gauges:
            y0, y1, x0, x1 = cy - r, cy + r, cx - r, cx + r
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx - cx, yy - cy
            rr = np.hypot(dx, dy)
            ring = (rr > r - 20) & (rr < r + 2) & ((dx < 20) | (dy > -20))  # evita el icono
            col = ring & (sat[y0:y1, x0:x1] > 60)
            ang = (np.degrees(np.arctan2(-dx, -dy)) + 360) % 360  # 0 arriba, antihorario
            amax = ang[col].max() if col.any() else 1
            track = np.median(s[y0:y1, x0:x1][ring & (sat[y0:y1, x0:x1] < 12) & (s[y0:y1, x0:x1].mean(-1) < 225)
                                              & (s[y0:y1, x0:x1].mean(-1) > 170)], axis=0)
            self.arcs.append(dict(box=(y0, y1, x0, x1), col=col, ang=ang, amax=amax, track=track))

    def cursor(self, t):
        return path_pos(self.PATH, t)

    def effects(self, img, t):
        p = self.cursor(t)
        # enlaces: subrayado
        for r in self.links.values():
            h = hover(self.PATH, r, t)
            if h > 0:
                y = r[3] - 1
                img[y:y + 1, r[0] + 14:r[2] - 4] = img[y:y + 1, r[0] + 14:r[2] - 4] * (1 - h) + ACCENT * h
        # pestañas: fondo resaltado + barra inferior
        for key, r in self.tabs.items():
            h = hover(self.PATH, r, t)
            sel = ease_io((t - self.CLICKS[0]) / 0.2) if key == "status" else 0
            if h > 0:
                reg = img[r[1]:r[3], r[0]:r[2]]
                reg[:] = reg * (1 - 0.07 * h) + ACCENT * 0.07 * h
            if h > 0 or sel > 0:
                a = max(h, sel)
                img[r[3] - 3:r[3], r[0]:r[2]] = img[r[3] - 3:r[3], r[0]:r[2]] * (1 - a) + ACCENT * a
        # indicadores: se redibujan tras el clic en Current Status
        if self.REDRAW <= t < self.REDRAW + 1.6:
            for i, g in enumerate(self.arcs):
                k = ease_io((t - self.REDRAW - 0.12 * i) / 1.1)
                y0, y1, x0, x1 = g["box"]
                hide = g["col"] & (g["ang"] > g["amax"] * k)
                img[y0:y1, x0:x1][hide] = g["track"]
        # hover sobre los indicadores: halo suave
        for (c, r, rect, _) in self.gauges:
            h = hover(self.PATH, rect, t)
            if h > 0:
                ov = np.zeros(img.shape[:2], np.float32)
                cv2.circle(ov, c, r + 10, 1.0, -1, cv2.LINE_AA)
                ov = cv2.GaussianBlur(ov, (0, 0), 8)[..., None] * 0.06 * h
                img[:] = img * (1 - ov) + ACCENT * ov
        return img

    def overlay(self, frame, A, k, t):
        """elementos dibujados a resolución de salida: tooltip y cursor"""
        for (c, r, rect, (t1, t2)) in self.gauges:
            h = hover(self.PATH, rect, t, lag=0.35)
            if h > 0:
                p = self.cursor(t)
                q = A @ np.array([p[0] + self.off[0], p[1] + self.off[1], 1.0])
                frame = tooltip(frame, q + [22, 26], t1, t2, h)
        p = self.cursor(t)
        q = A @ np.array([p[0] + self.off[0], p[1] + self.off[1], 1.0])
        press = max(0.0, 1 - min(abs(t - c) for c in self.CLICKS) / 0.12)
        frame = draw_arrow(frame, q, 1.05 * k ** 0.55, press)
        if self.CLICKS[0] <= t < self.CLICKS[0] + 0.6:
            frame = ripple(frame, q, (t - self.CLICKS[0]) / 0.6, 55 * k)
        return frame


class Movil:
    name = "app_movil"
    src = "captura_movil.jpg"
    DUR = 11.5
    pointer = "touch"

    def __init__(self):
        self.screen = np.array(Image.open(os.path.join(HERE, self.src)).convert("RGB"))
        h, w = self.screen.shape[:2]
        B, R = 30, 70  # bisel y radio del teléfono
        dev = np.zeros((h + 2 * B, w + 2 * B, 4), np.uint8)
        body = rounded_mask(w + 2 * B, h + 2 * B, R + B)
        dev[..., :3] = (22, 24, 28)
        dev[..., 3] = body
        # brillo sutil en el borde del bisel
        edge = cv2.morphologyEx(body, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        dev[..., :3] = np.where(edge[..., None] > 0, (70, 74, 82), dev[..., :3])
        sm = rounded_mask(w, h, R).astype(np.float32)[..., None] / 255
        dev[B:B + h, B:B + w, :3] = (self.screen * sm + dev[B:B + h, B:B + w, :3] * (1 - sm)).astype(np.uint8)
        self.dev, self.off, self.smask = dev, (B, B), sm
        cur = np.array
        self.TAPS = [  # (t, punto, tipo)
            (2.35, cur([656.0, 300.0]), "alertas"),
            (4.95, cur([450.0, 1030.0]), "conectar"),
            (8.05, cur([769.0, 1527.0]), "mant"),
            (9.75, cur([769.0, 1742.0]), "taller"),
        ]
        self.boxes = {"alertas": ("rect", (451, 183, 862, 393), 26), "mant": ("circle", (769, 1527), 58),
                      "taller": ("circle", (769, 1742), 58)}
        H = h + 2 * B
        fit = (OH - 90) / H
        cx = w / 2 + B
        self.CAM = [
            (0.0, fit, cx, H / 2), (1.0, fit, cx, H / 2),
            (2.6, 1.05, cx, 520), (4.0, 1.05, cx, 560),
            (5.2, 1.12, cx, 1000), (6.9, 1.15, cx, 1060),
            (7.9, 1.3, cx + 60, 1560), (9.2, 1.3, cx + 60, 1600),
            (self.DUR, 1.42, cx + 70, 1640),
        ]

    def effects(self, img, t):
        for tt, p, kind in self.TAPS:
            press = max(0.0, 1 - abs(t - tt) / 0.16)
            if kind == "conectar" and tt <= t < tt + 1.9:
                # ondas de conexión desde el botón central
                for j in range(3):
                    k = (t - tt - 0.35 * j) / 1.2
                    if 0 <= k < 1:
                        ov = np.zeros(img.shape[:2], np.float32)
                        cv2.circle(ov, (450, 1030), int(66 + 150 * k), 1.0, 6, cv2.LINE_AA)
                        ov = cv2.GaussianBlur(ov, (0, 0), 2)[..., None] * 0.55 * (1 - k)
                        img[:] = img * (1 - ov) + np.array([74, 102, 238], np.float32) * ov
            if press > 0 and kind in self.boxes:
                shape, g, r = self.boxes[kind]
                ov = np.zeros(img.shape[:2], np.float32)
                if shape == "rect":
                    cv2.rectangle(ov, g[:2], g[2:], 1.0, -1)
                else:
                    cv2.circle(ov, g, r, 1.0, -1, cv2.LINE_AA)
                ov = cv2.GaussianBlur(ov, (0, 0), 1.5)[..., None] * 0.16 * press
                img[:] = img * (1 - ov)
        return img

    def overlay(self, frame, A, k, t):
        for tt, p, kind in self.TAPS:
            a0, a1 = tt - 0.35, tt + 0.35  # el dedo aparece, pulsa y se levanta
            if a0 <= t <= a1:
                vis = min(1.0, (t - a0) / 0.12, (a1 - t) / 0.15)
                press = max(0.0, 1 - abs(t - tt) / 0.14)
                q = A @ np.array([p[0] + self.off[0], p[1] + self.off[1], 1.0])
                frame = draw_touch(frame, q, 34 * k, vis, press)
            if tt <= t < tt + 0.55:
                q = A @ np.array([p[0] + self.off[0], p[1] + self.off[1], 1.0])
                frame = ripple(frame, q, (t - tt) / 0.55, 70 * k)
        return frame


class Pantalla:
    """Base para capturas 16:9: la vista inicial llena el encuadre y la cámara se acerca."""
    DUR = 8.5
    pointer = "arrow"
    CURSOR_BOX = None  # (x0, y0, x1, y1) del cursor que trae la captura, para borrarlo

    def __init__(self):
        self.screen = np.array(Image.open(os.path.join(HERE, self.src)).convert("RGB"))
        if self.CURSOR_BOX:
            x0, y0, x1, y1 = self.CURSOR_BOX
            self.screen[y0:y1, x0:x1] = self.screen[y0 - 6, x0 - 6]
        h, w = self.screen.shape[:2]
        self.dev = np.dstack([self.screen, np.full((h, w), 255, np.uint8)])
        self.off = (0, 0)
        self.fit = OW / w
        self.base = self.screen.astype(np.float32)
        self.tips = []  # (rect, título, texto)

    def cursor(self, t):
        return path_pos(self.PATH, t)

    # --- efectos reutilizables (sobre la captura, en float)
    def tint(self, img, r, a, color=ACCENT):
        if a > 0:
            reg = img[r[1]:r[3], r[0]:r[2]]
            reg[:] = reg * (1 - a) + np.array(color, np.float32) * a

    def disc(self, img, c, rad, a, color=ACCENT):
        if a > 0:
            ov = np.zeros(img.shape[:2], np.float32)
            cv2.circle(ov, c, rad, 1.0, -1, cv2.LINE_AA)
            ov = ov[..., None] * a
            img[:] = img * (1 - ov) + np.array(color, np.float32) * ov

    def lift(self, img, r, h, press=0.0, M=18):
        x0, y0, x1, y1 = r
        if h <= 0 and press <= 0:
            return
        X0, Y0, X1, Y1 = x0 - M, y0 - M, x1 + M, y1 + M
        patch = self.base[Y0:Y1, X0:X1]
        ph, pw = patch.shape[:2]
        scale = 1 + 0.025 * h - 0.015 * press
        lift = -4 * h + 2 * press
        P = 50
        sh = np.zeros((ph + 2 * P, pw + 2 * P), np.float32)
        cv2.rectangle(sh, (M + P + 4, int(M + P + 10 + lift)), (x1 - X0 + P - 4, int(y1 - Y0 + P + 12 + lift)), 1.0, -1)
        sh = np.clip((cv2.GaussianBlur(sh, (0, 0), 11) - 0.03) / 0.97, 0, 1)
        reg = img[Y0 - P:Y1 + P, X0 - P:X1 + P]
        reg *= (1 - 0.18 * h * sh)[..., None]
        A = cv2.getRotationMatrix2D((pw / 2, ph / 2), 0, scale)
        A[1, 2] += lift
        warped = cv2.warpAffine(patch, A, (pw, ph), flags=cv2.INTER_CUBIC)
        m0 = np.zeros((ph, pw), np.float32)
        m0[M:M + y1 - y0, M:M + x1 - x0] = 1
        m = cv2.warpAffine(m0, A, (pw, ph), flags=cv2.INTER_LINEAR)[..., None]
        img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - m) + warped * m

    def overlay(self, frame, A, k, t):
        p = self.cursor(t)
        q = A @ np.array([p[0] + self.off[0], p[1] + self.off[1], 1.0])
        for rect, t1, t2 in self.tips:
            h = hover(self.PATH, rect, t, lag=0.35)
            if h > 0:
                frame = tooltip(frame, q + [22, 26], t1, t2, h)
        press = max([0.0] + [1 - abs(t - c) / 0.12 for c in self.CLICKS])
        frame = draw_arrow(frame, q, 1.05 * k ** 0.55, press)
        for c in self.CLICKS:
            if c <= t < c + 0.6:
                frame = ripple(frame, q, (t - c) / 0.6, 45 * k)
        return frame


def _rows(y0, pitch, n, x0, x1):
    return [(x0, int(y0 + i * pitch), x1, int(y0 + (i + 1) * pitch)) for i in range(n)]


class Operaciones(Pantalla):
    name = "operaciones"
    src = "captura_operaciones.webp"
    CURSOR_BOX = (686, 986, 707, 1013)
    DUR = 9.0

    def __init__(self):
        super().__init__()
        cur = np.array
        self.rows = _rows(514, 28, 6, 243, 1883)
        self.actions = [(1888, r[1] + 5, 1966, r[3] - 3) for r in self.rows]
        self.buttons = {"booking": (1373, 186, 1500, 220), "quotation": (1523, 186, 1662, 220),
                        "checkin": (1686, 186, 1813, 220), "invoice": (1837, 186, 1976, 220)}
        self.PATH = [
            (0.0, cur([696.0, 1000.0])), (0.8, cur([696.0, 1000.0])),
            (2.0, cur([720.0, 530.0])), (2.5, cur([720.0, 530.0])),
            (3.1, cur([560.0, 614.0])), (3.7, cur([560.0, 614.0])),
            (4.9, cur([1928.0, 616.0])), (5.5, cur([1928.0, 616.0])),
            (6.2, cur([1760.0, 206.0])), (6.6, cur([1760.0, 206.0])),
            (7.1, cur([1596.0, 206.0])), (self.DUR, cur([1596.0, 206.0])),
        ]
        self.CLICKS = [7.55]
        f = self.fit
        self.CAM = [
            (0.0, f, 1000, 562), (0.9, f, 1000, 562),
            (3.0, 1.3, 820, 600), (4.9, 1.4, 1400, 560),
            (6.8, 1.75, 1600, 330), (self.DUR, 1.95, 1620, 290),
        ]

    def effects(self, img, t):
        for r in self.rows:
            self.tint(img, r, 0.07 * hover(self.PATH, r, t))
        for r in self.actions:
            self.tint(img, r, 0.12 * hover(self.PATH, r, t), (60, 60, 70))
        for key, r in self.buttons.items():
            h = hover(self.PATH, r, t)
            press = max(0.0, 1 - abs(t - self.CLICKS[0]) / 0.12) if key == "quotation" else 0
            self.tint(img, (r[0] + 2, r[1] + 2, r[2] - 2, r[3] - 2), 0.14 * h + 0.1 * press, (70, 175, 110))
        return img


class Presupuesto(Pantalla):
    name = "nuevo_presupuesto"
    src = "captura_presupuesto.webp"
    CURSOR_BOX = (1236, 675, 1265, 716)
    DUR = 8.5

    def __init__(self):
        super().__init__()
        cur = np.array
        cols = [481, 562, 644, 727, 798, 871, 949]
        rows = [618, 645, 672, 700, 727]
        self.days = {(c, r): (c - 16, r - 13, c + 16, r + 13) for c in cols for r in rows}
        self.times = {lab: (1017, y - 14, 1548, y + 14) for lab, y in
                      zip(["10:30", "11:00", "11:30", "12:00", "12:30", "13:00"], [594, 623, 652, 681, 711, 740])}
        self.button = (1440, 790, 1563, 823)
        self.PATH = [
            (0.0, cur([1248.0, 696.0])), (0.7, cur([1248.0, 696.0])),
            (1.8, cur([646.0, 702.0])), (2.15, cur([646.0, 702.0])),
            (2.5, cur([729.0, 702.0])), (2.8, cur([729.0, 702.0])),
            (3.1, cur([800.0, 702.0])), (3.6, cur([800.0, 702.0])),
            (4.4, cur([1296.0, 654.0])), (4.7, cur([1296.0, 654.0])),
            (5.0, cur([1296.0, 683.0])), (5.6, cur([1296.0, 683.0])),
            (6.6, cur([1503.0, 808.0])), (self.DUR, cur([1503.0, 808.0])),
        ]
        self.CLICKS = [3.35, 5.3, 7.1]
        self.tips = [((1440, 790, 1563, 823), "Viernes 26 junio 2026 · 12:00", "Crear presupuesto")]
        f = self.fit
        self.CAM = [
            (0.0, f, 1000, 562), (0.7, f, 1000, 562),
            (2.4, 1.5, 760, 580), (3.6, 1.5, 900, 620),
            (5.2, 1.65, 1220, 650), (6.6, 1.8, 1330, 700),
            (self.DUR, 1.9, 1360, 715),
        ]

    def effects(self, img, t):
        for (c, r), rect in self.days.items():
            self.disc(img, (c, r), 15, 0.13 * hover(self.PATH, rect, t))
        sel = ease_io((t - self.CLICKS[1]) / 0.15)
        for lab, r in self.times.items():
            h = hover(self.PATH, r, t)
            a = 0.05 * h + (0.13 * sel if lab == "12:00" else 0)
            self.tint(img, r, a)
            if lab == "12:00" and sel > 0:
                self.tint(img, (r[0], r[1], r[0] + 4, r[3]), 0.9 * sel)
        h = hover(self.PATH, self.button, t)
        press = max(0.0, 1 - abs(t - self.CLICKS[2]) / 0.12)
        b = self.button
        self.tint(img, (b[0] + 2, b[1] + 2, b[2] - 2, b[3] - 2), 0.1 * h + 0.1 * press, (225, 70, 60))
        return img


class Recambios(Pantalla):
    name = "recambios_cesta"
    src = "captura_recambios.webp"
    CURSOR_BOX = (1570, 548, 1591, 576)
    DUR = 9.0
    NEW_ROW = ["Brakes", "TRW", "GDB1550", "00000001.00", "42.10", "10.00", "7.96", "45.85"]
    COLS = [1209, 1304, 1421, 1550, 1671, 1745, 1805, 1855]

    def __init__(self):
        super().__init__()
        cur = np.array
        xs = [(50, 224), (234, 407), (417, 590), (600, 774), (784, 957), (967, 1140)]
        self.cards = [(a, 282, b, 586) for a, b in xs] + [(a, 596, b, 873) for a, b in xs]
        self.link = (238, 411, 316, 425)
        self.finish = (559, 1067, 621, 1100)
        # fila nueva: plantilla a partir de una fila blanca (Filters), sin texto
        tpl = self.base[138:161].copy()
        tpl[:, 1204:1900] = 255
        pil = Image.fromarray(tpl.astype(np.uint8))
        d = ImageDraw.Draw(pil)
        fnt = ImageFont.truetype(FONT, 11)
        for txt, x in zip(self.NEW_ROW, self.COLS):
            d.text((x, 5), txt, fill=(80, 80, 85), font=fnt)
        self.row = np.array(pil).astype(np.float32)
        self.T_ADD = 3.45
        self.PATH = [
            (0.0, cur([1579.0, 562.0])), (0.6, cur([1579.0, 562.0])),
            (2.0, cur([330.0, 470.0])), (2.4, cur([330.0, 470.0])),
            (2.95, cur([280.0, 419.0])), (3.3, cur([280.0, 419.0])),
            (4.7, cur([1500.0, 197.0])), (5.6, cur([1500.0, 197.0])),
            (7.2, cur([592.0, 1085.0])), (self.DUR, cur([592.0, 1085.0])),
        ]
        self.CLICKS = [3.2, 7.65]
        self.tips = [((1203, 185, 1963, 208), "Añadido a la cesta", "Pastillas freno delanteras · 45,85 €")]
        f = self.fit
        self.CAM = [
            (0.0, f, 1000, 562), (0.6, f, 1000, 562),
            (2.2, 1.6, 420, 470), (3.3, 1.7, 400, 440),
            (4.8, 1.5, 1500, 300), (5.8, 1.55, 1520, 290),
            (7.3, 1.3, 800, 820), (self.DUR, 1.45, 700, 860),
        ]

    def effects(self, img, t):
        for r in self.cards:
            if r == self.cards[1]:
                continue
            self.lift(img, r, hover(self.PATH, r, t))
        self.lift(img, self.cards[1], hover(self.PATH, self.cards[1], t),
                  max(0.0, 1 - abs(t - self.CLICKS[0]) / 0.12))
        h = hover(self.PATH, self.link, t)
        if h > 0:
            r = self.link
            img[r[3]:r[3] + 1, r[0]:r[2]] = img[r[3]:r[3] + 1, r[0]:r[2]] * (1 - h) + ACCENT * h
        # la cesta se abre para la nueva fila
        if t >= self.T_ADD:
            k = ease_io((t - self.T_ADD) / 0.35)
            dy = int(round(23 * k))
            y0, y1 = 185, 345
            if dy:
                img[y0 + dy:y1 + dy, 1195:1975] = self.base[y0:y1, 1195:1975]
                row = self.row[:dy] if dy < 23 else self.row
                a = ease_io((t - self.T_ADD - 0.2) / 0.3)
                img[y0:y0 + len(row), 1195:1975] = (self.base[138:138 + len(row), 1195:1975] * (1 - a)
                                                     + row[:, 1195:1975] * a)
                flash = max(0.0, 1 - (t - self.T_ADD - 0.3) / 2.0) if t > self.T_ADD + 0.3 else a
                self.tint(img, (1204, y0, 1962, y0 + dy), 0.22 * flash, (90, 190, 120))
        b = self.finish
        press = max(0.0, 1 - abs(t - self.CLICKS[1]) / 0.12)
        self.tint(img, (b[0] + 2, b[1] + 2, b[2] - 2, b[3] - 2), 0.12 * hover(self.PATH, b, t) + 0.1 * press)
        return img


# ---------------------------------------------------------------- utilidades
def path_pos(PATH, t):
    for (ta, pa), (tb, pb) in zip(PATH, PATH[1:]):
        if ta <= t <= tb:
            k = ease_io((t - ta) / (tb - ta)) if tb > ta else 1.0
            d = pb - pa
            return pa + d * k + np.array([-d[1], d[0]]) * 0.07 * np.sin(np.pi * k)
    return PATH[-1][1]


def hover(PATH, r, t, lag=0.18):
    acc = 0.0
    for dt in np.linspace(-lag, 0, 6):
        p = path_pos(PATH, max(t + dt, 0))
        acc += float(r[0] - 3 <= p[0] <= r[2] + 3 and r[1] - 3 <= p[1] <= r[3] + 3)
    return acc / 6


def draw_arrow(frame, p, s, press):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil, "RGBA")
    s *= 1 - 0.12 * press
    pts = [(0, 0), (0, 17), (4.2, 13), (7, 19.5), (9.6, 18.4), (6.9, 12.2), (12.2, 12.2)]
    x, y = p
    d.polygon([(x + a * s + s, y + b * s + s * 1.3) for a, b in pts], fill=(0, 0, 0, 50))
    d.polygon([(x + a * s, y + b * s) for a, b in pts], fill=(255, 255, 255, 255))
    d.line([(x + a * s, y + b * s) for a, b in pts + [pts[0]]], fill=(20, 20, 20, 255), width=max(1, round(s)))
    return np.array(pil)


def draw_touch(frame, p, r, vis, press):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil, "RGBA")
    r *= 1 - 0.18 * press
    x, y = p
    d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int((90 + 60 * press) * vis)),
              outline=(60, 60, 70, int(150 * vis)), width=max(2, int(r / 12)))
    return np.array(pil)


def ripple(frame, p, k, rmax):
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil, "RGBA")
    r = 8 + rmax * k
    x, y = p
    d.ellipse([x - r, y - r, x + r, y + r], fill=(42, 110, 200, int(60 * (1 - k))))
    return np.array(pil)


F_T1 = ImageFont.truetype(FONT, 20)
F_T2 = ImageFont.truetype(FONTB, 24)


def tooltip(frame, q, t1, t2, a):
    pil = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    w = int(max(d.textlength(t1, font=F_T1), d.textlength(t2, font=F_T2))) + 36
    x, y = q
    x = min(x, OW - w - 20)
    d.rounded_rectangle([x + 4, y + 5, x + w + 4, y + 80], 10, fill=(0, 0, 0, int(55 * a)))
    d.rounded_rectangle([x, y, x + w, y + 75], 10, fill=(40, 44, 54, int(240 * a)))
    d.text((x + 18, y + 12), t1, fill=(190, 198, 212, int(255 * a)), font=F_T1)
    d.text((x + 18, y + 38), t2, fill=(255, 255, 255, int(255 * a)), font=F_T2)
    return np.array(Image.alpha_composite(pil, ov).convert("RGB"))


PAD = 140  # margen transparente alrededor para que la sombra no se corte


def camera(prof, t):
    k, cx, cy = interp(prof.CAM, t)
    H, W = prof.dev.shape[0] - 2 * PAD, prof.dev.shape[1] - 2 * PAD
    vw, vh = OW / k, OH / k
    if vw < W:
        cx = min(max(cx, vw / 2), W - vw / 2)
    if vh < H:
        cy = min(max(cy, vh / 2), H - vh / 2)
    cx, cy = cx + PAD, cy + PAD
    return k, np.float32([[k, 0, OW / 2 - cx * k], [0, k, OH / 2 - cy * k]])


def render(prof, t, shadow):
    scr = prof.effects(prof.screen.astype(np.float32).copy(), t)
    dev = prof.dev.astype(np.float32).copy()
    ox, oy = prof.off
    h, w = scr.shape[:2]
    if hasattr(prof, "smask"):
        dev[oy:oy + h, ox:ox + w, :3] = scr * prof.smask + dev[oy:oy + h, ox:ox + w, :3] * (1 - prof.smask)
    else:
        dev[oy:oy + h, ox:ox + w, :3] = scr
    k, A = camera(prof, t)
    interp_flag = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    if k < 1:  # reducir con buena calidad: primero se escala, luego se coloca
        small = cv2.resize(dev, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        B = np.float32([[1, 0, A[0, 2]], [0, 1, A[1, 2]]])
        rgba = cv2.warpAffine(small, B, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        sh = cv2.warpAffine(cv2.resize(shadow, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), B, (OW, OH))
    else:
        rgba = cv2.warpAffine(dev, A, (OW, OH), flags=interp_flag, borderValue=(0, 0, 0, 0))
        sh = cv2.warpAffine(shadow, A, (OW, OH), flags=cv2.INTER_LINEAR)
    a = np.clip(rgba[..., 3:] / 255, 0, 1)
    out = BACK * (1 - 0.22 * sh[..., None])
    out = out * (1 - a) + rgba[..., :3] * a
    out = np.clip(out, 0, 255).astype(np.uint8)
    if k > 1.05:
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        amt = min(0.55, (k - 1) * 0.5)
        out = cv2.addWeighted(out, 1 + amt, blur, -amt, 0)
    return prof.overlay(out, A, k, t)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "web"
    prof = {"web": Web, "movil": Movil, "operaciones": Operaciones, "presupuesto": Presupuesto,
            "recambios": Recambios}[which]()
    prof.dev = np.pad(prof.dev, ((PAD, PAD), (PAD, PAD), (0, 0)))
    prof.off = (prof.off[0] + PAD, prof.off[1] + PAD)
    alpha = prof.dev[..., 3].astype(np.float32) / 255
    shadow = cv2.GaussianBlur(np.roll(alpha, 22, axis=0), (0, 0), 28)
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        fs = [cv2.resize(render(prof, t, shadow), (960, 540), interpolation=cv2.INTER_AREA) for t in ts]
        while len(fs) % 2:
            fs.append(np.zeros_like(fs[0]))
        grid = np.vstack([np.hstack(fs[i:i + 2]) for i in range(0, len(fs), 2)])
        cv2.imwrite(os.path.join(HERE, "_p.png"), grid[:, :, ::-1])
        return
    out = os.path.join(HERE, prof.name + ".mp4")
    w = imageio_ffmpeg.write_frames(out, (OW, OH), fps=FPS, codec="libx264", quality=None, macro_block_size=1,
                                    output_params=["-crf", "17", "-preset", "slow", "-movflags", "+faststart"])
    w.send(None)
    for i in range(int(prof.DUR * FPS)):
        w.send(np.ascontiguousarray(render(prof, i / FPS, shadow)))
    w.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.system(f'"{ff}" -y -loglevel error -i "{out}" -vf "fps=12,scale=720:-1:flags=lanczos,split[a][b];'
              f'[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:'
              f'diff_mode=rectangle" "{os.path.join(HERE, prof.name + ".gif")}"')


if __name__ == "__main__":
    main()
