"""Timeline for the 224 Gb/s PAM-4 PCB-dispersion animation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .pcb_physics import (
    PAM4_SYMBOLS,
    eye_traces,
    pam4_transmit_waveform,
    pam4_waveform,
    pcb_channel,
    pcb_group_delay,
)
from .rendering import (
    AMBER,
    BACKGROUND,
    CORAL,
    CYAN,
    GREEN,
    GRID,
    MUTED,
    PANEL,
    TEXT,
    VIOLET,
    Canvas,
    PlotBox,
    clamp,
    crossfade,
    ease,
    fade_color,
)


DURATION = 27.0
SAMPLES_PER_SYMBOL = 64


def _node(canvas: Canvas, box: tuple[int, int, int, int], title: str, detail: str, accent: tuple[int, int, int]) -> None:
    canvas.rect(box, PANEL, accent)
    canvas.text((box[0] + 22, box[1] + 34), title, 26, TEXT, bold=True)
    canvas.text((box[0] + 22, box[1] + 76), detail, 18, MUTED)


def _eye_y(box: PlotBox, value: float) -> float:
    return box.bottom - (value + 4.4) / 8.8 * box.height


def _draw_eye(
    canvas: Canvas,
    box: PlotBox,
    traces: list[np.ndarray],
    color: tuple[int, int, int],
    alpha: float,
    *,
    label_openings: bool = False,
) -> None:
    canvas.plot_axes(box)
    x = np.linspace(0.0, 2.0, traces[0].size)
    for threshold in (-2.0, 0.0, 2.0):
        y = _eye_y(box, threshold)
        canvas.dashed_line(
            (box.left, y),
            (box.right, y),
            fade_color(AMBER, 0.28),
            1.3,
            dash=10,
            gap=10,
        )
    for trace in traces[:40]:
        canvas.plot_curve(box, x, trace, fade_color(color, alpha), y_min=-4.4, y_max=4.4, width=2)
    center_x = 0.5 * (box.left + box.right)
    canvas.dashed_line((center_x, box.top), (center_x, box.bottom), fade_color(CYAN, 0.7), 2)
    canvas.text((center_x, box.top - 18), "最佳采样", 17, CYAN, anchor="ms")

    if not label_openings:
        return
    bracket_x = center_x + 34
    for lower, upper, label in ((1.0, 3.0, "上眼"), (-1.0, 1.0, "中眼"), (-3.0, -1.0, "下眼")):
        top = _eye_y(box, upper - 0.28)
        bottom = _eye_y(box, lower + 0.28)
        canvas.line([(bracket_x, top), (bracket_x, bottom)], GREEN, 2.5)
        canvas.line([(bracket_x - 8, top), (bracket_x + 8, top)], GREEN, 2.5)
        canvas.line([(bracket_x - 8, bottom), (bracket_x + 8, bottom)], GREEN, 2.5)
        canvas.text((bracket_x + 15, 0.5 * (top + bottom)), label, 17, GREEN, bold=True, anchor="lm")


def render_intro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.2)
    canvas.text((110, 86), "场景三 · 高速数字电路", 24, fade_color(CYAN, reveal), bold=True)
    canvas.text((960, 390), "224 Gb/s PAM-4 信号的 PCB 传输", 56, fade_color(TEXT, reveal), bold=True, anchor="mm")
    canvas.text((960, 478), "封装与走线 · 频率相关群延迟 · 接收眼图", 29, fade_color(MUTED, reveal), anchor="mm")
    line = clamp((local_time - 0.4) / 1.1)
    canvas.line([(510, 558), (510 + 900 * line, 558)], fade_color(AMBER, line), 4)
    canvas.text((960, 650), "高速边沿经过真实互连后，为何出现振铃和眼图闭合？", 28, fade_color(TEXT, line), anchor="mm")
    return canvas.image


def render_link(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("01  真实系统位置", "高速 SerDes 通道", "失真在封装、过孔和 PCB 走线中累积，在接收均衡器之前观察")
    boxes = (
        (90, 320, 385, 440, "PAM-4 发射机", "112 GBaud · 四个电平", CYAN),
        (505, 320, 800, 440, "芯片封装与过孔", "阻抗不连续与寄生效应", VIOLET),
        (920, 320, 1215, 440, "PCB 微带线", "介质与结构产生色散", AMBER),
        (1335, 320, 1830, 440, "接收均衡与判决", "FFE / DFE 恢复采样裕量", CORAL),
    )
    for left, top, right, bottom, title, detail, accent in boxes:
        _node(canvas, (left, top, right, bottom), title, detail, accent)
    for start, end in (((385, 380), (505, 380)), ((800, 380), (920, 380)), ((1215, 380), (1335, 380))):
        canvas.arrow(start, end, GRID, 4)
    phase = (max(local_time, 0.0) / 3.2) % 1.0
    canvas.circle((395 + 925 * phase, 380), 10, AMBER)

    canvas.text((960, 520), "色散观察点", 23, CORAL, bold=True, anchor="mm")
    canvas.arrow((1215, 505), (1325, 505), CORAL, 3)

    levels = (3, 1, -1, -3)
    for index, level in enumerate(levels):
        y = 650 + index * 78
        canvas.line([(270, y), (610, y)], fade_color(GRID, 0.9), 2)
        canvas.text((220, y), f"{level:+d}", 22, MUTED, anchor="rm", latin=True)
    symbols = (3, 1, -1, -3, -1, 3, 1)
    points: list[tuple[float, float]] = []
    for index, symbol in enumerate(symbols):
        x0 = 270 + index * 48
        x1 = x0 + 48
        y = 650 + (3 - symbol) / 2 * 78
        points.extend(((x0, y), (x1, y)))
    canvas.line(points, CYAN, 5)
    canvas.text((900, 755), "2 bit / symbol", 26, CYAN, bold=True)
    canvas.formula((900, 820), "224 Gb/s ÷ 2 = 112 GBaud", 31, TEXT)
    return canvas.image


def render_channel(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("02  通道色散", "高频分量获得不同群延迟", "真实 PCB 还伴随导体与介质损耗；此处用两条曲线分开表达")
    progress = ease(clamp(local_time / 6.0))
    frequency = np.linspace(0.0, 1.0, 500)
    group_delay = pcb_group_delay(frequency, 2.6 * progress)
    magnitude = np.exp(-1.1 * progress * np.power(frequency, 1.35))
    left = PlotBox(125, 390, 900, 820)
    right = PlotBox(1020, 390, 1795, 820)
    canvas.text((left.left, left.top - 58), "群延迟", 28, AMBER, bold=True)
    canvas.text((right.left, right.top - 58), "幅度响应", 28, VIOLET, bold=True)
    canvas.plot_axes(left)
    canvas.plot_axes(right)
    canvas.plot_curve(left, frequency, np.asarray(group_delay), AMBER, y_min=0.0, y_max=5.5, width=5)
    canvas.plot_curve(right, frequency, magnitude, VIOLET, y_min=0.0, y_max=1.1, width=5)
    canvas.formula((left.left + 24, left.top + 28), "τ_{g}(ω) = −dφ/dω", 29, AMBER)
    canvas.formula((right.left + 24, right.top + 28), "|H(ω)|", 29, VIOLET)
    canvas.text((left.right, left.bottom + 30), "归一化频率", 20, MUTED, anchor="ra")
    canvas.text((right.right, right.bottom + 30), "归一化频率", 20, MUTED, anchor="ra")
    canvas.text((960, 920), "相位色散改变到达时刻；高频损耗削弱陡峭边沿", 28, TEXT, bold=True, anchor="mm")
    canvas.formula((960, 988), "ε_{eff} = ε_{eff}(ω)", 31, CYAN, anchor="mm")
    return canvas.image


def render_waveform(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("03  时域后果", "边沿变缓并出现振铃", "灰色为发射参考，彩色为均衡前的接收波形")
    progress = ease(clamp(local_time / 6.0))
    symbols = np.tile(PAM4_SYMBOLS, 4)
    transmit = pam4_waveform(symbols, SAMPLES_PER_SYMBOL)
    received = pcb_channel(
        transmit,
        dispersion_strength=420.0 * progress,
        loss_strength=3.0 * progress,
    )
    start = 10 * SAMPLES_PER_SYMBOL
    stop = start + 10 * SAMPLES_PER_SYMBOL
    x = np.linspace(0.0, 10.0, stop - start)
    box = PlotBox(120, 345, 1800, 825)
    canvas.plot_axes(box)
    for level in (-3, -1, 1, 3):
        y = box.bottom - (level + 4.5) / 9.0 * box.height
        canvas.dashed_line((box.left, y), (box.right, y), fade_color(GRID, 0.75), 1.5)
    canvas.plot_curve(box, x, transmit[start:stop], fade_color(MUTED, 0.65), y_min=-4.5, y_max=4.5, width=3)
    canvas.plot_curve(box, x, received[start:stop], AMBER, y_min=-4.5, y_max=4.5, width=5)
    canvas.text((box.left - 24, box.top + 10), "幅度", 21, MUTED, anchor="ra")
    canvas.text((box.right, box.bottom + 30), "符号时间", 21, MUTED, anchor="ra")
    canvas.text((960, 910), "过冲与振铃侵入相邻符号的采样窗口", 30, CORAL, bold=True, anchor="mm")
    canvas.formula((960, 985), "R_{b} = 224 Gb/s    R_{s} = 112 GBaud", 30, TEXT, anchor="mm")
    return canvas.image


def render_eye(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("04  接收端后果", "PAM-4 眼图逐渐闭合", "横向虚线为三条判决门限；竖向虚线为最佳采样时刻")
    progress = ease(clamp(local_time / 5.8))
    rng = np.random.default_rng(224)
    symbols = rng.choice(np.array((-3.0, -1.0, 1.0, 3.0)), size=96)
    transmit = pam4_transmit_waveform(symbols, SAMPLES_PER_SYMBOL, bandwidth_per_baud=0.6)
    received = pcb_channel(
        transmit,
        dispersion_strength=4000.0 * progress,
        loss_strength=20.0 * progress,
    )
    ideal_traces = eye_traces(transmit, SAMPLES_PER_SYMBOL)
    received_traces = eye_traces(received, SAMPLES_PER_SYMBOL)
    left = PlotBox(125, 380, 900, 830)
    right = PlotBox(1020, 380, 1795, 830)
    canvas.text((left.left, left.top - 56), "发射眼图", 28, CYAN, bold=True)
    canvas.text((right.left, right.top - 56), "接收眼图 · 均衡前", 28, CORAL, bold=True)
    _draw_eye(canvas, left, ideal_traces, CYAN, 0.56, label_openings=True)
    _draw_eye(canvas, right, received_traces, CORAL, 0.46)
    canvas.text((512, 905), "三个采样窗口均有明显眼高", 25, GREEN, bold=True, anchor="mm")
    canvas.text((1408, 905), "垂直与水平裕量同时减小", 25, CORAL, bold=True, anchor="mm")
    canvas.text((960, 990), "FFE / DFE 在接收端补偿通道，但不能替代良好的互连设计", 26, TEXT, anchor="mm")
    return canvas.image


def render_outro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.0)
    canvas.text((960, 275), "高速互连中的色散来源", 32, fade_color(TEXT, reveal), bold=True, anchor="mm")
    _node(canvas, (310, 385, 880, 565), "材料色散", "介电常数随频率变化", fade_color(VIOLET, reveal))
    _node(canvas, (1040, 385, 1610, 565), "波导色散", "场分布与有效折射率随频率变化", fade_color(CYAN, reveal))
    canvas.arrow((880, 475), (1040, 475), fade_color(GRID, reveal), 4)
    canvas.text((960, 700), "共同结果：频率相关群延迟 → 波形畸变 → 眼图闭合", 34, fade_color(AMBER, reveal), bold=True, anchor="mm")
    canvas.formula((960, 850), "τ_{g}(ω) = −dφ(ω)/dω", 36, fade_color(TEXT, reveal), anchor="mm")
    canvas.text((960, 970), "为什么不同 PCB 结构需要不同的均衡策略？", 27, fade_color(TEXT, reveal), anchor="mm")
    return canvas.image


def render_frame(width: int, height: int, seconds: float) -> Image.Image:
    starts = (0.0, 2.2, 7.8, 14.0, 20.3, 25.0)
    renderers = (render_intro, render_link, render_channel, render_waveform, render_eye, render_outro)
    overlap = 0.55
    index = max(i for i, start in enumerate(starts) if start <= seconds)
    current = renderers[index](width, height, seconds - starts[index])
    if index == 0:
        return current
    amount = (seconds - starts[index]) / overlap
    if amount >= 1.0:
        return current
    previous = renderers[index - 1](width, height, seconds - starts[index - 1])
    blank = Image.new("RGB", (width, height), BACKGROUND)
    return crossfade(previous, blank, amount * 2.0) if amount < 0.5 else crossfade(blank, current, (amount - 0.5) * 2.0)
