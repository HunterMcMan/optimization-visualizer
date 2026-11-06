"""Render a narrated 9:16 showcase video of the Optimization Visualizer.

    pip install -r requirements.txt -r requirements-dev.txt
    python scripts/make_video.py            # -> media/showcase.mp4
    python scripts/make_video.py --preview  # contact sheets in build/video_preview/

The video drives the real app headlessly: it builds the landing page and the tools
with their build() functions, presses their buttons, drags their sliders and types
into their text boxes, rendering every frame with matplotlib's Agg backend. A
neural voice (edge-tts) narrates each scene, and subtitles are timed from the
voice's word boundaries. Frames and audio are encoded with the ffmpeg binary that
ships with imageio-ffmpeg.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app  # noqa: E402  (needs ROOT on sys.path)

warnings.filterwarnings("ignore")  # "FigureCanvasAgg is non-interactive" on fig.show()

W, H, FPS = 1080, 1920, 30
DPI = 150                 # app figures are rendered at this dpi, then scaled
VOICE = "en-US-BrianNeural"
FALLBACK_VOICE = "Microsoft David Desktop"   # offline Windows voice if edge-tts fails
LEAD, PAD = 0.35, 0.8     # seconds of quiet before and after each narration clip
CACHE = ROOT / "build" / "video_cache"
OUT = ROOT / "media" / "showcase.mp4"
PREVIEW_DIR = ROOT / "build" / "video_preview"

BAND_H = 150              # top band with the logo and the scene title
SLOT_X, SLOT_W = 20, 1040
SLOT_A_Y, SLOT_A_MAX = 170, 760
SLOT_B_BOTTOM = 1625      # the subtitles live below this
TITLE_BAR = 34
SUB_Y = 1770              # center of the subtitle box
ZOOM_COLOR = (255, 127, 14)  # tab:orange, the outline of the magnified area

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(str(FONTS / f"{name}{'-Bold' if bold else ''}.ttf"), size)


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------
# Narration

SCENES = [
    ("intro", "Optimization Visualizer",
     "This is Optimization Visualizer: interactive math for analysis and optimization. "
     "Pick a tool, and it opens in its own window."),
    ("epsilon_delta", "ε-δ limits",
     "First, epsilon delta limits. Drag delta, and the graph turns red when it's too wide "
     "for this epsilon. Shrink epsilon, and the largest working delta shrinks with it. "
     "With two variables you get a 3D view, and here the limit depends on the direction, "
     "so it doesn't exist."),
    ("lagrange", "Lagrange multipliers",
     "Next, Lagrange multipliers. Slide the level curve until it just touches the "
     "constraint. That's where the gradient of f lines up with the gradient of g. The tests "
     "window checks the second order condition two ways, and tells you if each point is a "
     "global or local extremum, or neither."),
    ("kkt", "KKT conditions",
     "Now add inequality constraints with the KKT conditions. The close-up shows the cone "
     "of active gradients. At a KKT point, minus grad f points inside it. If a multiplier "
     "goes negative, the arrow turns red. And the constraint qualifications window shows "
     "why some minimizers, like this cusp, aren't KKT points at all."),
    ("outro", "Optimization Visualizer",
     "It all ships as one app. Run python app dot py, or just open the exe. Give it a try!"),
]

# Spoken phrases shown differently in the subtitles.
PHRASES = [(("app", "dot", "py"), "app.py"), (("minus", "grad", "f"), "−∇f"),
           (("epsilon", "delta"), "ε-δ")]


def _audio_seconds(path):
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
                         capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


async def _edge(text, mp3):
    import edge_tts
    words = []
    communicate = edge_tts.Communicate(text, VOICE, boundary="WordBoundary")
    with open(mp3, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append([chunk["offset"] / 1e7, (chunk["offset"] + chunk["duration"]) / 1e7,
                              chunk["text"]])
    return words


def _windows_voice(text, wav):
    """Offline fallback: a built-in Windows voice, with evenly spread word timings."""
    script = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SelectVoice('{FALLBACK_VOICE}'); $s.SetOutputToWaveFile('{wav}'); "
              f"$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", script], input=text, text=True,
                   check=True)
    tokens = text.split()
    total = _audio_seconds(wav)
    weights = np.cumsum([0] + [len(t) + 2 for t in tokens])
    times = weights / weights[-1] * total
    return [[times[i], times[i + 1], t.strip(".,!?:;")] for i, t in enumerate(tokens)]


def narrate(text):
    """(audio path, duration, words as [start, end, text]) for one clip, cached by text."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16]
    meta = CACHE / f"{key}.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        return Path(data["audio"]), data["seconds"], data["words"]
    try:
        audio = CACHE / f"{key}.mp3"
        words = asyncio.run(_edge(text, audio))
    except Exception as err:
        print(f"edge-tts failed ({err}); using the offline Windows voice")
        audio = CACHE / f"{key}.wav"
        words = _windows_voice(text, audio)
    seconds = _audio_seconds(audio)
    meta.write_text(json.dumps({"audio": str(audio), "seconds": seconds, "words": words}),
                    encoding="utf-8")
    return audio, seconds, words


def subtitle_chunks(words, start):
    """Group words into short subtitle lines, timed in absolute seconds."""
    merged, i = [], 0
    while i < len(words):
        for phrase, shown in PHRASES:
            n = len(phrase)
            if tuple(w[2].lower().strip(".,!?") for w in words[i:i + n]) == phrase:
                merged.append([words[i][0], words[i + n - 1][1], shown])
                i += n
                break
        else:
            merged.append(list(words[i]))
            i += 1
    chunks, current = [], []
    for k, w in enumerate(merged):
        gap = k and w[0] - merged[k - 1][1] > 0.3
        sentence = current and current[-1][2][-1:] in ".,!?:" and len(current) >= 2
        if current and (len(current) >= 5 or gap or sentence):
            chunks.append(current)
            current = []
        current.append(w)
    if current:
        chunks.append(current)
    for k in range(len(chunks) - 1, 0, -1):  # no lone words: merge them backwards
        if len(chunks[k]) == 1 and len(chunks[k - 1]) <= 5:
            chunks[k - 1] += chunks.pop(k)
    out = []
    for k, chunk in enumerate(chunks):
        t0 = start + chunk[0][0]
        t1 = start + (chunks[k + 1][0][0] if k + 1 < len(chunks) else chunk[-1][1] + 0.4)
        out.append((t0, t1, " ".join(w[2] for w in chunk)))
    return out


# ---------------------------------------------------------------------------
# Windows and frames

class Win:
    """An app figure rendered to an image, re-rendered only after it changes."""

    def __init__(self, fig):
        self.fig = fig
        fig.set_dpi(DPI)
        self.version = 0
        fig.canvas.mpl_connect("draw_event", lambda _: self._bump())
        self._img, self._seen = None, -1

    def _bump(self):
        self.version += 1

    @property
    def title(self):
        return self.fig.canvas.manager.get_window_title()

    def image(self):
        if self.fig.stale or self._img is None:
            self.fig.canvas.draw()
        if self._seen != self.version or self._img is None:
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            self._img = Image.fromarray(np.ascontiguousarray(buf[..., :3]))
            self._seen = self.version
        return self._img


def newest_window():
    return Win(plt.figure(max(plt.get_fignums())))


def _fit(aspect, x, y, w, h):
    """Largest (x, y, w, h) with this aspect (w/h) centered in the box."""
    if w / h > aspect:
        nw = h * aspect
        return x + (w - nw) / 2, y, nw, h
    nh = w / aspect
    return x, y + (h - nh) / 2, w, nh


def _text_center(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def _wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def _logo(size):
    import io
    return Image.open(io.BytesIO(app.logo_png(size))).convert("RGBA")


def title_card(lines, logo=520, y_logo=360):
    """A full-frame white card with the logo and centered text lines (text, size, bold,
    color, mono)."""
    img = Image.new("RGBA", (W, H), "white")
    if logo:
        img.alpha_composite(_logo(logo), ((W - logo) // 2, y_logo))
    draw = ImageDraw.Draw(img)
    y = y_logo + logo + 60
    for text, size, bold, color, *mono in lines:
        fnt = font(size, bold, bool(mono))
        for line in _wrap(draw, text, fnt, W - 140):
            _text_center(draw, W / 2, y, line, fnt, color)
            y += int(size * 1.3)
        y += int(size * 0.5)
    return img


class Director:
    """Holds what is on screen and writes frames to the encoder (or preview sheets)."""

    def __init__(self, writer, preview):
        self.writer, self.preview = writer, preview
        self.frame = 0
        self.card = None          # full-frame image (title cards), overrides the slots
        self.a = None             # Win shown in the top slot
        self.b = None             # Win in the bottom slot, or None for the magnifier
        self.zoom = None          # magnified area of slot A, in figure fractions (y up)
        self.title = ""
        self.cursor = None        # (x, y) on the canvas
        self.pulse = None         # (x, y, start frame) of the last click
        self.subs = []
        self.thumbs = []
        self.scene_start = 0.0
        self.words = []
        self._band_cache = {}

    @property
    def t(self):
        return self.frame / FPS

    # -- geometry ---------------------------------------------------------
    def rect_a(self):
        img = self.a.image()
        x, y, w, h = _fit(img.width / (img.height + TITLE_BAR * img.width / SLOT_W),
                          SLOT_X, SLOT_A_Y, SLOT_W, SLOT_A_MAX)
        return x, y, w, h

    def content(self, rect):
        x, y, w, h = rect
        return x, y + TITLE_BAR, w, h - TITLE_BAR

    def to_canvas(self, fx, fy):
        """Figure fraction of the slot-A window -> canvas pixels."""
        x, y, w, h = self.content(self.rect_a())
        return x + fx * w, y + (1 - fy) * h

    def widget(self, ax):
        b = ax.get_position()
        return self.to_canvas((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)

    # -- drawing ----------------------------------------------------------
    def _band(self):
        if self.title not in self._band_cache:
            band = Image.new("RGBA", (W, BAND_H), "white")
            band.alpha_composite(_logo(104), (30, 22))
            draw = ImageDraw.Draw(band)
            draw.text((158, 26), "OPTIMIZATION VISUALIZER", font=font(24, True), fill="#888888")
            draw.text((156, 60), self.title, font=font(52, True), fill="black")
            draw.line([(30, BAND_H - 6), (W - 30, BAND_H - 6)], fill="#d3d3d3", width=2)
            self._band_cache[self.title] = band
        return self._band_cache[self.title]

    def _window(self, canvas, img, rect, title, crop=None):
        x, y, w, h = (int(round(v)) for v in rect)
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle([x + 6, y + 9, x + w + 6, y + h + 9], 10,
                                                 fill=(0, 0, 0, 45))
        canvas.alpha_composite(shadow)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle([x, y, x + w, y + h], 10, fill="#ebebeb", outline="#c8c8c8",
                               width=2)
        for i, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            draw.ellipse([x + 14 + 22 * i, y + 11, x + 26 + 22 * i, y + 23], fill=color)
        _text_center(draw, x + w / 2, y + 7, title, font(19, True), "#444444")
        if crop is not None:
            img = img.crop(crop)
        cx, cy, cw, ch = (int(round(v)) for v in self.content((x, y, w, h)))
        canvas.paste(img.resize((cw - 4, ch - 2), Image.LANCZOS).convert("RGBA"), (cx + 2, cy))

    def compose(self):
        if self.card is not None:
            canvas = self.card.copy()
        else:
            canvas = Image.new("RGBA", (W, H), "white")
            canvas.alpha_composite(self._band(), (0, 0))
            if self.a is not None:
                img = self.a.image()
                ra = self.rect_a()
                self._window(canvas, img, ra, self.a.title)
                bottom = ra[1] + ra[3] + 40
                if self.b is not None:
                    bi = self.b.image()
                    rb = _fit(bi.width / (bi.height + TITLE_BAR * bi.width / SLOT_W),
                              SLOT_X, bottom, SLOT_W, SLOT_B_BOTTOM - bottom)
                    self._window(canvas, bi, rb, self.b.title)
                elif self.zoom is not None:
                    x0, y0, x1, y1 = self.zoom
                    box = (x0 * img.width, (1 - y1) * img.height,
                           x1 * img.width, (1 - y0) * img.height)
                    aspect = (box[2] - box[0]) / (box[3] - box[1])
                    rb = _fit(aspect, SLOT_X, bottom, SLOT_W,
                              SLOT_B_BOTTOM - bottom - TITLE_BAR)
                    rb = (rb[0], rb[1], rb[2], rb[3] + TITLE_BAR)
                    self._window(canvas, img, rb, "magnified", crop=tuple(int(v) for v in box))
                    p0, p1 = self.to_canvas(x0, y1), self.to_canvas(x1, y0)
                    ImageDraw.Draw(canvas).rounded_rectangle([*p0, *p1], 6, outline=ZOOM_COLOR,
                                                             width=4)
        self._draw_cursor(canvas)
        self._draw_subtitle(canvas)
        return canvas

    def _draw_cursor(self, canvas):
        draw = ImageDraw.Draw(canvas)
        if self.pulse is not None:
            px, py, f0 = self.pulse
            u = (self.frame - f0) / (0.45 * FPS)
            if 0 <= u <= 1:
                r = 12 + 34 * u
                overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                ImageDraw.Draw(overlay).ellipse([px - r, py - r, px + r, py + r],
                                                outline=(31, 119, 180, int(220 * (1 - u))),
                                                width=6)
                canvas.alpha_composite(overlay)
        if self.cursor is not None:
            x, y = self.cursor
            shape = [(0, 0), (0, 40), (10, 30), (17, 46), (24, 43), (17, 28), (31, 28)]
            draw.polygon([(x + a, y + b) for a, b in shape], fill="white", outline="black",
                         width=3)

    def _draw_subtitle(self, canvas):
        text = next((s for t0, t1, s in self.subs if t0 <= self.t < t1), None)
        if not text:
            return
        draw = ImageDraw.Draw(canvas)
        fnt = font(48, True)
        lines = _wrap(draw, text, fnt, W - 160)
        line_h = 62
        box_h = line_h * len(lines) + 36
        box_w = max(draw.textlength(line, font=fnt) for line in lines) + 60
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(
            [W / 2 - box_w / 2, SUB_Y - box_h / 2, W / 2 + box_w / 2, SUB_Y + box_h / 2], 22,
            fill=(20, 20, 20, 215))
        canvas.alpha_composite(overlay)
        for i, line in enumerate(lines):
            _text_center(draw, W / 2, SUB_Y - box_h / 2 + 16 + i * line_h, line, fnt, "white")

    # -- output -----------------------------------------------------------
    def emit(self, image=None):
        if self.preview:
            if self.frame % 10 == 0:
                img = image if image is not None else self.compose()
                self.thumbs.append(img.convert("RGB").resize((W // 4, H // 4)))
        else:
            img = image if image is not None else self.compose()
            self.writer.send(img.convert("RGB").tobytes())
        self.frame += 1

    # -- timeline helpers ---------------------------------------------------
    def frames(self, seconds):
        return max(1, int(round(seconds * FPS)))

    def hold(self, seconds):
        for _ in range(self.frames(seconds)):
            self.emit()

    def until(self, scene_time):
        while self.t < self.scene_start + scene_time:
            self.emit()

    def cue(self, word):
        """Scene time at which the narration says `word`."""
        for start, _, text in self.words:
            if word.lower() in text.lower():
                return LEAD + start
        raise KeyError(word)

    def transition(self, change, seconds=0.5):
        """Cross-fade from the current picture to the one after change()."""
        before = self.compose().convert("RGB")
        change()
        n = self.frames(seconds)
        for i in range(n):
            after = self.compose().convert("RGB")
            self.emit(Image.blend(before, after, ease((i + 1) / n)).convert("RGBA"))

    def move(self, target, seconds=0.6):
        if self.cursor is None:
            self.cursor = (W * 0.6, H * 0.55)
        x0, y0 = self.cursor
        n = self.frames(seconds)
        for i in range(n):
            u = ease((i + 1) / n)
            self.cursor = (x0 + (target[0] - x0) * u, y0 + (target[1] - y0) * u)
            self.emit()

    def click(self, ax, action, move=0.6):
        self.move(self.widget(ax), move)
        self.pulse = (*self.cursor, self.frame)
        action()
        self.hold(0.3)

    def press(self, button, move=0.6):
        self.click(button.ax, lambda: button._observers.process("clicked", None), move)

    def sweep(self, slider, v0, v1, seconds):
        """Drag a slider from v0 to v1, with the cursor on its handle."""
        def handle(v):
            b = slider.ax.get_position()
            frac = (v - slider.valmin) / (slider.valmax - slider.valmin)
            return self.to_canvas(b.x0 + frac * (b.x1 - b.x0), (b.y0 + b.y1) / 2)
        slider.set_val(v0)
        self.move(handle(v0), 0.5)
        n = self.frames(seconds)
        for i in range(n):
            v = v0 + (v1 - v0) * ease((i + 1) / n)
            if i % 2 == 0 or i == n - 1:  # the app re-renders on every value: 15 per second
                slider.set_val(v)
            self.cursor = handle(v)
            self.emit()

    def type(self, box, text, seconds=1.2):
        """Type into a TextBox character by character, then press Enter once."""
        self.move(self.widget(box.ax), 0.5)
        original = box.text
        n = self.frames(seconds)
        shown = -1
        for i in range(n):
            k = int(round(len(text) * (i + 1) / n))
            if k != shown and i % 2 == 0:
                box.text_disp.set_text(text[:k])
                shown = k
            self.emit()
        box.text_disp.set_text(original)  # set_val ignores text the box already shows
        box.set_val(text)
        self.hold(0.2)

    def zoom_to(self, rect, seconds=0.6):
        start = self.zoom or rect
        n = self.frames(seconds)
        for i in range(n):
            u = ease((i + 1) / n)
            self.zoom = tuple(a + (b - a) * u for a, b in zip(start, rect))
            self.emit()

    def begin(self, name, title, text):
        audio, seconds, words = narrate(text)
        self.scene_start = self.t
        self.words = words
        self.subs += subtitle_chunks(words, self.t + LEAD)
        return audio, seconds + LEAD + PAD


# ---------------------------------------------------------------------------
# Scenes. Each gets the director and its length D (narration plus padding); it
# returns when its part of the video is done.

# The magnifier always shows a tight close-up of what is happening, never the whole
# window, so it adds detail instead of repeating the top slot. Boxes are figure
# fractions (x0, y0, x1, y1), computed from the app's own axes.

def data_box(ax, x0, x1, y0, y1):
    """Figure-fraction box around a rectangle given in an axes' data coordinates."""
    fig = ax.figure
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
    return (px0 / fig.bbox.width, py0 / fig.bbox.height,
            px1 / fig.bbox.width, py1 / fig.bbox.height)


def axes_box(ax, inset=0.0, pad=0.0):
    """Figure-fraction box of an axes as drawn, shrunk by `inset` or grown by `pad`."""
    b = ax.get_position(original=False)
    dx, dy = b.width * inset - pad, b.height * inset - pad
    return (b.x0 + dx, b.y0 + dy, b.x1 - dx, b.y1 - dy)


def card_box(button):
    """The landing-page card around an Open button (its rounded panel)."""
    b = button.ax.get_position()
    return (b.x0 - 0.016, 0.05, b.x1 + 0.016, 0.58)


def scene_intro(d, D, s):
    card = title_card([("Optimization", 96, True, "black"), ("Visualizer", 96, True, "black"),
                       ("Interactive math for analysis and optimization", 42, False,
                        "#666666")], y_logo=470)
    white = Image.new("RGBA", (W, H), "white")
    n = d.frames(1.0)
    for i in range(n):
        d.emit(Image.blend(white, card, ease((i + 1) / n)))
    d.card = card
    d.until(0.42 * D)
    s["landing"] = Win(app.build())
    buttons = s["landing"].fig._keep

    def show_landing():
        d.card, d.a, d.b, d.title = None, s["landing"], None, "Pick a tool"
        d.zoom = card_box(buttons[0])
    d.transition(show_landing, 0.6)
    for i, b in enumerate(buttons):  # the magnifier follows the cursor from card to card
        if i:
            d.zoom_to(card_box(b), 0.5)
        d.move(d.widget(b.ax), 0.6)
        d.hold(0.3)
    d.until(D)


def open_tool(d, s, index, title, zoom_for):
    """Click a card's Open button on the landing page and switch to the new window.
    `zoom_for(win)` gives the magnified box in the new window."""
    button = s["landing"].fig._keep[index]

    def back():
        d.a, d.b, d.zoom, d.title = s["landing"], None, card_box(button), title
    d.transition(back, 0.35)
    d.press(button, 0.5)
    win = newest_window()
    win.image()  # draw once, so axes positions and transforms are final
    box = zoom_for(win)

    def show():
        d.a, d.b, d.zoom = win, None, box
    d.transition(show, 0.5)
    return win


def scene_epsilon_delta(d, D, s):
    def around_a(win, dx, dy):
        view = win.fig._keep[-1]["view"]
        (a,), L = view.prob.a, view.prob.L
        return data_box(view.ax, a - dx, a + dx, L - dy, L + dy)

    win = open_tool(d, s, 0, "ε-δ limits", lambda w: around_a(w, 2.0, 1.75))
    tb_f, tb_a, s_eps, s_delta, *_ = win.fig._keep
    d.sweep(s_delta, 0.5, 1.7, max(1.0, d.cue("red") + 0.6 - (d.t - d.scene_start)))
    d.hold(0.6)
    d.sweep(s_delta, 1.7, 0.6, 0.9)
    d.until(d.cue("shrink") - 0.4)
    d.zoom_to(around_a(win, 0.85, 0.75), 0.6)  # move in as epsilon gets small
    d.sweep(s_eps, 1.0, 0.35, 1.2)
    d.sweep(s_delta, 0.6, 0.3, 0.7)
    d.until(d.cue("variables") - 1.2)
    tb_a.text_disp.set_text("0, 0")
    d.type(tb_f, "x*y/(x^2+y^2)", 1.3)
    view = win.fig._keep[-1]["view"]
    win.image()
    d.zoom_to(axes_box(view.ax3, inset=0.06), 0.6)  # the 3D plot alone, while it turns
    elev, azim = view.ax3.elev, view.ax3.azim
    n = d.frames(max(1.0, D - (d.t - d.scene_start)))
    for i in range(n):
        if i % 2 == 0:
            view.ax3.view_init(elev, azim + 70 * ease(i / n))
        d.emit()


def scene_lagrange(d, D, s):
    def tangency(win):
        view = win.fig._keep[-1]["view"]
        return data_box(view.ax2, -0.15, 1.55, -0.15, 1.55)  # the max on the unit circle

    win = open_tool(d, s, 1, "Lagrange multipliers", tangency)
    keep = win.fig._keep
    tb_f, tb_g, b_next, b_tests, state = keep[0], keep[1], keep[4], keep[5], keep[-1]
    k = state["k"]
    d.until(d.cue("slide") - 0.2)
    d.sweep(k, k.val, np.sqrt(2), max(1.0, d.cue("touches") + 0.4 - (d.t - d.scene_start)))
    d.until(d.cue("tests") - 0.6)
    d.press(b_tests, 0.5)
    tests = Win(state["tests"].fig)
    d.transition(lambda: setattr(d, "b", tests), 0.5)
    d.hold(0.8)
    d.press(b_next, 0.5)
    d.until(d.cue("neither") - 1.9)
    tb_g.text_disp.set_text("y = x^3")
    d.type(tb_f, "y", 0.5)
    d.until(D)


def scene_kkt(d, D, s):
    # The close-up panel is always centered on the selected point, so its middle
    # stays the right place to magnify when the example changes.
    win = open_tool(d, s, 2, "KKT conditions",
                    lambda w: axes_box(w.fig._keep[-1]["view"].axz, inset=0.12))
    buttons, state = win.fig._keep[3], win.fig._keep[-1]
    d.until(d.cue("negative") - 1.8)
    d.press(buttons["example"], 0.5)
    d.press(buttons["example"], 0.3)
    d.press(buttons["next"], 0.5)
    d.until(d.cue("qualifications") - 0.4)
    d.press(buttons["cqs"], 0.5)
    cqs = Win(state["cqs"].fig)
    d.transition(lambda: setattr(d, "b", cqs), 0.5)
    d.until(d.cue("cusp") - 1.0)
    d.press(buttons["example"], 0.5)
    d.until(D)


def scene_outro(d, D, s):
    card = title_card([("Optimization Visualizer", 64, True, "black"),
                       ("python app.py", 50, False, "#1f77b4", True),
                       ("or open OptimizationVisualizer.exe", 38, False, "#444444"),
                       ("Built with NumPy · SciPy · SymPy · Matplotlib", 30, False, "#888888")],
                      logo=420, y_logo=420)

    def show():
        d.card, d.cursor = card, None
    d.transition(show, 0.7)
    d.until(D)


SCENE_FUNCS = {"intro": scene_intro, "epsilon_delta": scene_epsilon_delta,
               "lagrange": scene_lagrange, "kkt": scene_kkt, "outro": scene_outro}


# ---------------------------------------------------------------------------

def mux(video, clips, out):
    """Place each narration clip at its scene start, mix, and add the track to the video."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video)]
    filters = []
    for i, (audio, start) in enumerate(clips, 1):
        args += ["-i", str(audio)]
        ms = int(round(start * 1000))
        filters.append(f"[{i}:a]adelay=delays={ms}:all=1,aresample=48000[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(clips) + 1))
    filters.append(f"{mix}amix=inputs={len(clips)}:normalize=0:duration=longest[aout]")
    args += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(out)]
    subprocess.run(args, check=True)


def contact_sheets(thumbs, starts):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    bounds = [int(s * FPS / 10) for s in starts] + [len(thumbs)]
    for (name, *_), lo, hi in zip(SCENES, bounds, bounds[1:]):
        sel = thumbs[lo:hi][::max(1, (hi - lo) // 12)][:12]
        cols = 6
        rows = (len(sel) + cols - 1) // cols
        tw, th = W // 4, H // 4
        sheet = Image.new("RGB", (cols * tw + (cols + 1) * 8, rows * th + (rows + 1) * 8),
                          "#333333")
        for i, img in enumerate(sel):
            sheet.paste(img, (8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + 8)))
        path = PREVIEW_DIR / f"{len(list(PREVIEW_DIR.glob('*.png'))):02d}_{name}.png"
        sheet.save(path)
        print("preview:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", action="store_true",
                        help="sample every 10th frame into contact sheets instead of a video")
    args = parser.parse_args()

    silent = OUT.with_name("showcase_silent.mp4")
    writer = None
    if not args.preview:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(str(silent), (W, H), fps=FPS, codec="libx264",
                                             macro_block_size=8, pix_fmt_in="rgb24",
                                             output_params=["-crf", "20", "-preset", "medium"])
        writer.send(None)
    d = Director(writer, args.preview)
    clips, starts, state = [], [], {}
    for name, title, text in SCENES:
        audio, D = d.begin(name, title, text)
        d.title = title
        starts.append(d.t)
        clips.append((audio, d.t + LEAD))
        print(f"{name}: starts {d.t:.1f}s, narration {D - LEAD - PAD:.1f}s", flush=True)
        SCENE_FUNCS[name](d, D, state)
    print(f"{d.frame} frames, {d.t:.1f}s")
    if args.preview:
        for f in list(PREVIEW_DIR.glob("*.png")) if PREVIEW_DIR.exists() else []:
            f.unlink()
        contact_sheets(d.thumbs, starts)
        return
    writer.close()
    mux(silent, clips, OUT)
    silent.unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
