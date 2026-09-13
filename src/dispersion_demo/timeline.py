"""Scene composition and the presentation timeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .physics import (
    gaussian_intensity,
    gaussian_pulse_train,
    group_velocity,
    phase_velocity,
    two_frequency_wave,
)
from .rendering import (
    AMBER,
    BACKGROUND,
    CORAL,
    CYAN,
    GREEN,
    GRID,
    MUTED,
    TEXT,
    VIOLET,
    WHITE,
    Canvas,
    PlotBox,
    clamp,
    crossfade,
    ease,
    fade_color,
)


DURATION = 36.0


def _wave_panel(
    canvas: Canvas,
    box: PlotBox,
    x: np.ndarray,
    values: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    alpha: float = 1.0,
) -> None:
    center = 0.5 * (box.top + box.bottom)
    canvas.line([(box.left, center), (box.right, center)], fade_color(GRID, alpha), 1.5)
    canvas.plot_curve(
        box,
        x,
        values,
        fade_color(color, alpha),
        y_min=-1.25,
        y_max=1.25,
        width=3,
    )
    canvas.formula(
        (box.left + 14, box.top + 16),
        label,
        24,
        fade_color(color, alpha),
        bold=True,
    )


def render_intro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.2)
    canvas.text((110, 86), "电磁场与电磁波", 24, fade_color(CYAN, reveal), bold=True)
    canvas.text((960, 385), "从复合波到脉冲展宽", 62, fade_color(TEXT, reveal), bold=True, anchor="mm")
    canvas.text(
        (960, 470),
        "相速度 · 群速度 · 系统色散",
        30,
        fade_color(MUTED, reveal),
        anchor="mm",
    )
    line_reveal = clamp((local_time - 0.55) / 1.25)
    canvas.line([(540, 552), (540 + 840 * line_reveal, 552)], fade_color(AMBER, line_reveal), 4)
    canvas.text(
        (960, 634),
        "两个相近频率的波，如何形成一个传播的包络？",
        28,
        fade_color(TEXT, clamp((local_time - 1.0) / 1.0)),
        anchor="mm",
    )
    return canvas.image


def render_beats(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("01  双频叠加", "包络如何产生", "教材图的动态复现")
    x = np.linspace(0.0, 18.0, 1000)
    model_time = max(0.0, local_time) * 0.72
    wave1, wave2, total, envelope = two_frequency_wave(x, model_time)

    top1 = PlotBox(130, 282, 905, 445)
    top2 = PlotBox(1015, 282, 1790, 445)
    bottom = PlotBox(130, 535, 1790, 860)
    _wave_panel(canvas, top1, x, wave1, CYAN, "E_{1} = E_{0} cos(ω_{1}t − β_{1}z)")
    _wave_panel(canvas, top2, x, wave2, CORAL, "E_{2} = E_{0} cos(ω_{2}t − β_{2}z)", clamp((local_time - 0.8) / 0.8))

    sum_alpha = clamp((local_time - 1.7) / 0.9)
    center = 0.5 * (bottom.top + bottom.bottom)
    canvas.line([(bottom.left, center), (bottom.right, center)], fade_color(GRID, sum_alpha), 2)
    canvas.plot_curve(
        bottom,
        x,
        total,
        fade_color(WHITE, sum_alpha),
        y_min=-2.25,
        y_max=2.25,
        width=4,
    )
    env_abs = np.abs(envelope)
    canvas.plot_curve(
        bottom,
        x,
        env_abs,
        fade_color(AMBER, sum_alpha),
        y_min=-2.25,
        y_max=2.25,
        width=3,
    )
    canvas.plot_curve(
        bottom,
        x,
        -env_abs,
        fade_color(AMBER, sum_alpha),
        y_min=-2.25,
        y_max=2.25,
        width=3,
    )
    canvas.formula(
        (bottom.left + 14, bottom.top + 12),
        "E = E_{1} + E_{2}",
        26,
        fade_color(WHITE, sum_alpha),
        bold=True,
    )
    canvas.formula((bottom.right - 12, bottom.bottom + 28), "z", 23, MUTED, anchor="rm")

    annotation = clamp((local_time - 3.2) / 1.0)
    vp = phase_velocity(5.5)
    vg = group_velocity()
    canvas.arrow((245, 945), (245 + 235 * vp, 945), fade_color(CYAN, annotation), 4)
    canvas.text((500, 945), "相速度", 23, fade_color(CYAN, annotation), anchor="lm")
    canvas.formula((615, 945), "v_{p}", 25, fade_color(CYAN, annotation))
    canvas.arrow((960, 945), (960 + 235 * vg, 945), fade_color(AMBER, annotation), 4)
    canvas.text((1215, 945), "群速度", 23, fade_color(AMBER, annotation), anchor="lm")
    canvas.formula((1330, 945), "v_{g} = dω/dβ", 25, fade_color(AMBER, annotation))
    return canvas.image


def render_bridge(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("02  从两个频率到有限带宽", "波包不是单一频率", "频率越丰富，时域包络越局域")
    progress = ease(local_time / 3.0)
    canvas.text((420, 300), "频率域", 28, CYAN, bold=True, anchor="ma")
    canvas.text((1440, 300), "时域", 28, AMBER, bold=True, anchor="ma")
    canvas.line([(960, 285), (960, 895)], GRID, 2)

    spec = PlotBox(170, 365, 850, 780)
    canvas.plot_axes(spec)
    canvas.formula((spec.right, spec.bottom + 30), "ω", 23, MUTED, anchor="rm")
    frequencies = np.linspace(-3.0, 3.0, 25)
    amplitudes = np.exp(-0.5 * np.square(frequencies / 1.18))
    for index, (frequency, amplitude) in enumerate(zip(frequencies, amplitudes)):
        density_reveal = clamp((progress * 25 - abs(index - 12)) / 4.0)
        if index in (10, 14):
            density_reveal = 1.0
        x_pos = spec.left + (frequency + 3.0) / 6.0 * spec.width
        top = spec.bottom - amplitude * 310 * density_reveal
        color = CYAN if frequency < 0 else CORAL
        canvas.line([(x_pos, spec.bottom), (x_pos, top)], fade_color(color, 0.9), 5)

    wave = PlotBox(1060, 365, 1780, 780)
    z = np.linspace(-4.0, 4.0, 900)
    envelope_width = 2.25 - 1.3 * progress
    packet = np.exp(-np.square(z / envelope_width)) * np.cos(10.0 * z)
    envelope = np.exp(-np.square(z / envelope_width))
    center = 0.5 * (wave.top + wave.bottom)
    canvas.line([(wave.left, center), (wave.right, center)], GRID, 2)
    canvas.plot_curve(wave, z, packet, WHITE, y_min=-1.2, y_max=1.2, width=4)
    canvas.plot_curve(wave, z, envelope, AMBER, y_min=-1.2, y_max=1.2, width=3)
    canvas.plot_curve(wave, z, -envelope, AMBER, y_min=-1.2, y_max=1.2, width=3)

    canvas.text((960, 910), "2 个离散频率", 25, CYAN, bold=True, anchor="ra")
    canvas.arrow((990, 910), (1180, 910), MUTED, 3)
    canvas.text((1210, 910), "连续频谱构成局域波包", 25, AMBER, bold=True, anchor="la")
    return canvas.image


def render_broadening(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("03  二阶色散", "同样的传播距离，不同的波包结果", "纵轴统一为光强；只观察色散引起的形状变化")
    progress = ease(clamp(local_time / 8.8))

    canvas.text((110, 270), "传播距离", 22, MUTED)
    canvas.line([(275, 282), (1785, 282)], GRID, 5)
    marker_x = 275 + 1510 * progress
    canvas.circle((marker_x, 282), 9, AMBER)
    canvas.formula((marker_x, 316), f"L/L_{{max}} = {progress:0.2f}", 21, AMBER, anchor="mm")

    left = PlotBox(125, 410, 900, 835)
    right = PlotBox(1020, 410, 1795, 835)
    t = np.linspace(-4.5, 4.5, 900)
    initial = gaussian_intensity(t)
    unchanged = gaussian_intensity(t, dispersion_strength=0.0)
    strength = 3.1 * progress
    broadened = gaussian_intensity(t, dispersion_strength=strength)

    for box, title, formula, accent in (
        (left, "无色散", "β_{2} = 0", CYAN),
        (right, "有色散", "β_{2} ≠ 0", AMBER),
    ):
        canvas.text((box.left, box.top - 62), title, 28, accent, bold=True)
        canvas.formula((box.left + 175, box.top - 47), formula, 28, accent, bold=True)
        canvas.plot_axes(box)
        canvas.formula((box.left - 22, box.top + 6), "I", 23, MUTED, anchor="rm")
        canvas.text((box.right, box.bottom + 28), "迟延时间 T", 20, MUTED, anchor="ra")
        canvas.plot_curve(box, t, initial, fade_color(MUTED, 0.65), y_min=0.0, y_max=1.12, width=3)

    canvas.fill_curve(left, t, unchanged, CYAN, y_min=0.0, y_max=1.12)
    canvas.plot_curve(left, t, unchanged, CYAN, y_min=0.0, y_max=1.12, width=5)
    canvas.fill_curve(right, t, broadened, AMBER, y_min=0.0, y_max=1.12)
    canvas.plot_curve(right, t, broadened, AMBER, y_min=0.0, y_max=1.12, width=5)

    canvas.text((512, 910), "仅整体平移，形状不变", 23, CYAN, bold=True, anchor="ma")
    canvas.text((1408, 910), "宽度增加 · 峰值下降 · 能量不变", 23, AMBER, bold=True, anchor="ma")
    canvas.formula(
        (960, 1008),
        "β(ω_{0} + Ω) ≈ β_{0} + β_{1}Ω + 1/2 β_{2}Ω^{2}",
        28,
        TEXT,
        anchor="mm",
    )
    return canvas.image


def render_isi(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("04  系统后果", "展宽脉冲污染相邻码元", "OOK 示例：在码元中心采样，观察判决值如何被前后码元改变")
    progress = ease(clamp(local_time / 5.7))
    symbol_period = 1.8
    symbols = (0, 1, 0, 1, 1, 0, 1, 0)
    centers = tuple((index - 3.5) * symbol_period for index in range(len(symbols)))
    visible_symbols = symbols[1:7]
    visible_centers = centers[1:7]
    left_edge = visible_centers[0] - symbol_period / 2.0
    right_edge = visible_centers[-1] + symbol_period / 2.0
    t = np.linspace(left_edge, right_edge, 1400)
    box = PlotBox(120, 375, 1800, 810)
    canvas.plot_axes(box)
    canvas.formula((box.left - 22, box.top + 6), "I_{R}", 23, MUTED, anchor="rm")
    canvas.text((box.right, box.bottom + 30), "时间", 21, MUTED, anchor="ra")

    strength = 4.8 * progress
    components, received = gaussian_pulse_train(
        t,
        symbols,
        centers,
        tau0=0.40,
        dispersion_strength=strength,
    )
    receiver_gain = 1.0 / max(float(np.max(received)), 1e-12)
    components *= receiver_gain
    received *= receiver_gain

    threshold = 0.5
    threshold_y = box.bottom - threshold / 1.12 * box.height
    canvas.dashed_line(
        (box.left, threshold_y),
        (box.right, threshold_y),
        fade_color(AMBER, 0.7),
        2,
        dash=14,
        gap=10,
    )
    canvas.text((box.right - 10, threshold_y - 12), "判决门限", 18, AMBER, anchor="rs")

    for boundary_index in range(7):
        boundary = left_edge + boundary_index * symbol_period
        x_pos = box.left + (boundary - t[0]) / (t[-1] - t[0]) * box.width
        canvas.line([(x_pos, box.top), (x_pos, box.bottom)], fade_color(GRID, 0.55), 1.5)

    contribution_colors = (CYAN, VIOLET, AMBER, GREEN)
    active_index = 0
    for symbol, component in zip(symbols, components):
        if symbol == 0:
            continue
        color = contribution_colors[active_index % len(contribution_colors)]
        active_index += 1
        canvas.plot_curve(box, t, component, fade_color(color, 0.48), y_min=0.0, y_max=1.12, width=2.5)

    canvas.fill_curve(box, t, received, CORAL, y_min=0.0, y_max=1.12)
    canvas.plot_curve(box, t, received, CORAL, y_min=0.0, y_max=1.12, width=5)

    sample_values = np.interp(np.asarray(visible_centers), t, received)
    error_count = 0
    for bit, center, sample in zip(visible_symbols, visible_centers, sample_values):
        x_pos = box.left + (center - t[0]) / (t[-1] - t[0]) * box.width
        y_pos = box.bottom - sample / 1.12 * box.height
        detected = int(sample >= threshold)
        is_error = detected != bit
        error_count += int(is_error)
        sample_color = CORAL if is_error else GREEN
        canvas.dashed_line((x_pos, box.top), (x_pos, box.bottom), fade_color(CYAN, 0.42), 1.5)
        canvas.circle((x_pos, y_pos), 8, sample_color)
        canvas.text((x_pos, box.top - 58), str(bit), 30, TEXT, bold=True, anchor="mm", latin=True)
        canvas.text((x_pos, box.top - 25), "采样", 16, CYAN, anchor="mm")
        if is_error:
            canvas.text((x_pos, box.bottom + 58), "0 → 1", 20, CORAL, bold=True, anchor="mm", latin=True)

    canvas.line([(138, 280), (198, 280)], fade_color(CYAN, 0.6), 3)
    canvas.text((215, 280), "各个 1 码元的展宽贡献", 18, MUTED, anchor="lm")
    canvas.line([(535, 280), (595, 280)], CORAL, 5)
    canvas.text((612, 280), "叠加后的接收波形", 18, TEXT, anchor="lm")

    if error_count:
        conclusion = "0 码元被相邻脉冲尾部抬过门限，发生误判"
        conclusion_color = CORAL
    else:
        conclusion = "脉冲尾部逐渐进入相邻采样点，判决裕量下降"
        conclusion_color = AMBER
    canvas.text((960, 930), conclusion, 27, conclusion_color, bold=True, anchor="mm")
    canvas.formula((960, 995), "I_{R}(t) = ∑ a_{k}p(t − kT)", 29, TEXT, anchor="mm")
    return canvas.image


def render_outro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.0)
    canvas.text((960, 320), "频率相关的群时延", 30, fade_color(CYAN, reveal), anchor="mm")
    canvas.arrow((960, 382), (960, 500), fade_color(MUTED, reveal), 4)
    canvas.text((960, 575), "波包展宽", 58, fade_color(AMBER, reveal), bold=True, anchor="mm")
    canvas.arrow((960, 642), (960, 760), fade_color(MUTED, reveal), 4)
    canvas.text((960, 836), "码间干扰", 38, fade_color(CORAL, reveal), bold=True, anchor="mm")
    canvas.text((960, 982), "为什么不同频率的群时延不同？", 25, fade_color(TEXT, reveal), anchor="mm")
    return canvas.image


def render_frame(width: int, height: int, seconds: float) -> Image.Image:
    """Render one frame using short overlaps between adjacent scenes."""

    scene_starts = (0.0, 2.2, 12.6, 15.6, 26.6, 33.4)
    renderers = (render_intro, render_beats, render_bridge, render_broadening, render_isi, render_outro)
    overlap = 0.6

    current_index = max(index for index, start in enumerate(scene_starts) if start <= seconds)
    current = renderers[current_index](width, height, seconds - scene_starts[current_index])
    if current_index == 0:
        return current
    transition_progress = (seconds - scene_starts[current_index]) / overlap
    if transition_progress >= 1.0:
        return current
    previous = renderers[current_index - 1](
        width,
        height,
        seconds - scene_starts[current_index - 1],
    )
    blank = Image.new("RGB", (width, height), BACKGROUND)
    if transition_progress < 0.5:
        return crossfade(previous, blank, transition_progress * 2.0)
    return crossfade(blank, current, (transition_progress - 0.5) * 2.0)
