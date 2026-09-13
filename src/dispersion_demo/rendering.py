"""Reusable Pillow drawing primitives for the scientific animation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RGB = tuple[int, int, int]

BACKGROUND: RGB = (7, 15, 26)
PANEL: RGB = (12, 25, 40)
GRID: RGB = (43, 62, 80)
TEXT: RGB = (233, 240, 246)
MUTED: RGB = (150, 166, 181)
CYAN: RGB = (44, 200, 231)
AMBER: RGB = (255, 190, 61)
CORAL: RGB = (255, 112, 101)
GREEN: RGB = (87, 210, 156)
VIOLET: RGB = (177, 145, 255)
WHITE: RGB = (248, 251, 253)

CHINESE_FONT = Path("/System/Library/Fonts/STHeiti Medium.ttc")
CHINESE_LIGHT_FONT = Path("/System/Library/Fonts/STHeiti Light.ttc")
LATIN_FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
LATIN_BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
MATH_ROMAN_FONT = Path("/System/Library/Fonts/Supplemental/STIXGeneral.otf")
MATH_ITALIC_FONT = Path("/System/Library/Fonts/Supplemental/STIXGeneralItalic.otf")
MATH_BOLD_FONT = Path("/System/Library/Fonts/Supplemental/STIXGeneralBol.otf")


@lru_cache(maxsize=64)
def load_font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    if latin:
        path = LATIN_BOLD_FONT if bold else LATIN_FONT
    else:
        path = CHINESE_FONT if bold else CHINESE_LIGHT_FONT
    return ImageFont.truetype(str(path), size=size)


@lru_cache(maxsize=64)
def load_math_font(size: int, italic: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = MATH_BOLD_FONT if bold else MATH_ITALIC_FONT if italic else MATH_ROMAN_FONT
    return ImageFont.truetype(str(path), size=size)


def formula_runs(value: str) -> list[tuple[str, int]]:
    """Parse a small math-markup subset into baseline, subscript and superscript runs."""

    runs: list[tuple[str, int]] = []
    buffer: list[str] = []
    index = 0

    def flush() -> None:
        if buffer:
            runs.append(("".join(buffer), 0))
            buffer.clear()

    while index < len(value):
        marker = value[index]
        if marker not in "_^":
            buffer.append(marker)
            index += 1
            continue
        flush()
        script = -1 if marker == "_" else 1
        index += 1
        if index >= len(value):
            break
        if value[index] == "{":
            end = value.find("}", index + 1)
            if end == -1:
                raise ValueError(f"Unclosed formula script in {value!r}")
            content = value[index + 1 : end]
            index = end + 1
        else:
            content = value[index]
            index += 1
        runs.append((content, script))
    flush()
    return runs


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def fade_color(color: RGB, alpha: float, background: RGB = BACKGROUND) -> RGB:
    alpha = clamp(alpha)
    return tuple(round(background[i] + alpha * (color[i] - background[i])) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class PlotBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.scale = min(width / 1920.0, height / 1080.0)
        self.image = Image.new("RGB", (width, height), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

    def px(self, value: float) -> int:
        return round(value * self.scale)

    def font(self, size: float, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
        return load_font(max(10, self.px(size)), bold=bold, latin=latin)

    def text(
        self,
        xy: tuple[float, float],
        value: str,
        size: float,
        color: RGB = TEXT,
        *,
        bold: bool = False,
        anchor: str = "la",
        latin: bool = False,
    ) -> None:
        self.draw.text(
            (self.px(xy[0]), self.px(xy[1])),
            value,
            font=self.font(size, bold=bold, latin=latin),
            fill=color,
            anchor=anchor,
            spacing=self.px(8),
        )

    def formula(
        self,
        xy: tuple[float, float],
        value: str,
        size: float,
        color: RGB = TEXT,
        *,
        anchor: str = "lm",
        bold: bool = False,
    ) -> None:
        """Draw STIX math text with independently positioned scripts.

        Use ``x_{sub}`` and ``x^{sup}`` in ``value``. The coordinate is the
        visual center line; horizontal anchors follow Pillow's l/m/r prefix.
        """

        base_size = max(10, self.px(size))
        script_size = max(8, round(base_size * 0.68))
        parsed = formula_runs(value)
        pieces: list[tuple[str, int, ImageFont.FreeTypeFont]] = []
        upright_words = ("cos", "sin", "exp", "ln", "max", "eff")

        for text, script in parsed:
            font_size = script_size if script else base_size
            cursor = 0
            while cursor < len(text):
                upright = next((word for word in upright_words if text.startswith(word, cursor)), None)
                if upright:
                    pieces.append((upright, script, load_math_font(font_size, bold=bold)))
                    cursor += len(upright)
                    continue
                char = text[cursor]
                italic = char.isalpha() and char not in "d"
                font = load_math_font(font_size, italic=italic, bold=bold and not italic)
                if pieces and pieces[-1][1] == script and pieces[-1][2] == font:
                    previous, previous_script, _ = pieces[-1]
                    pieces[-1] = (previous + char, previous_script, font)
                else:
                    pieces.append((char, script, font))
                cursor += 1

        widths = [self.draw.textlength(text, font=font) for text, _, font in pieces]
        total_width = sum(widths)
        x = self.px(xy[0])
        if anchor.startswith("m"):
            x -= total_width / 2
        elif anchor.startswith("r"):
            x -= total_width

        base_font = load_math_font(base_size)
        ascent, descent = base_font.getmetrics()
        baseline = self.px(xy[1]) + (ascent - descent) / 2
        for (text, script, font), width in zip(pieces, widths):
            offset = base_size * (0.27 if script == -1 else -0.43 if script == 1 else 0.0)
            self.draw.text((round(x), round(baseline + offset)), text, font=font, fill=color, anchor="ls")
            x += width

    def line(
        self,
        points: Iterable[tuple[float, float]],
        color: RGB,
        width: float = 2.0,
    ) -> None:
        self.draw.line(
            [(self.px(x), self.px(y)) for x, y in points],
            fill=color,
            width=max(1, self.px(width)),
            joint="curve",
        )

    def dashed_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: RGB,
        width: float = 2.0,
        dash: float = 12.0,
        gap: float = 9.0,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        cursor = 0.0
        while cursor < length:
            stop = min(length, cursor + dash)
            self.line(
                [(x1 + cursor * ux, y1 + cursor * uy), (x1 + stop * ux, y1 + stop * uy)],
                color,
                width,
            )
            cursor += dash + gap

    def rect(self, box: tuple[float, float, float, float], fill: RGB, outline: RGB | None = None) -> None:
        scaled = tuple(self.px(v) for v in box)
        self.draw.rectangle(scaled, fill=fill, outline=outline, width=self.px(2))

    def circle(self, center: tuple[float, float], radius: float, fill: RGB) -> None:
        x, y = center
        self.draw.ellipse(
            (self.px(x - radius), self.px(y - radius), self.px(x + radius), self.px(y + radius)),
            fill=fill,
        )

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: RGB,
        width: float = 4.0,
    ) -> None:
        self.line([start, end], color, width)
        x1, y1 = start
        x2, y2 = end
        angle = np.arctan2(y2 - y1, x2 - x1)
        head = 16.0
        spread = 0.52
        p1 = (x2 - head * np.cos(angle - spread), y2 - head * np.sin(angle - spread))
        p2 = (x2 - head * np.cos(angle + spread), y2 - head * np.sin(angle + spread))
        self.line([p1, end, p2], color, width)

    def plot_axes(self, box: PlotBox, color: RGB = GRID) -> None:
        self.line([(box.left, box.bottom), (box.right, box.bottom)], color, 2)
        self.line([(box.left, box.top), (box.left, box.bottom)], color, 2)

    def plot_curve(
        self,
        box: PlotBox,
        x: np.ndarray,
        y: np.ndarray,
        color: RGB,
        *,
        y_min: float,
        y_max: float,
        width: float = 3.0,
    ) -> None:
        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y_min) / (y_max - y_min)
        px = box.left + x_norm * box.width
        py = box.bottom - y_norm * box.height
        self.line(zip(px.tolist(), py.tolist()), color, width)

    def fill_curve(
        self,
        box: PlotBox,
        x: np.ndarray,
        y: np.ndarray,
        color: RGB,
        *,
        y_min: float,
        y_max: float,
    ) -> None:
        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y_min) / (y_max - y_min)
        px = box.left + x_norm * box.width
        py = box.bottom - y_norm * box.height
        points = [(self.px(box.left), self.px(box.bottom))]
        points.extend((self.px(float(a)), self.px(float(b))) for a, b in zip(px, py))
        points.append((self.px(box.right), self.px(box.bottom)))
        overlay = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).polygon(points, fill=(*color, 38))
        self.image = Image.alpha_composite(self.image.convert("RGBA"), overlay).convert("RGB")
        self.draw = ImageDraw.Draw(self.image)

    def header(self, step: str, title: str, subtitle: str = "") -> None:
        self.text((110, 76), step, 23, CYAN, bold=True)
        self.text((110, 125), title, 48, TEXT, bold=True)
        if subtitle:
            self.text((110, 183), subtitle, 24, MUTED)
        self.line([(110, 224), (1810, 224)], GRID, 2)


def crossfade(first: Image.Image, second: Image.Image, amount: float) -> Image.Image:
    return Image.blend(first, second, ease(amount))
