"""Timeline for the wideband-radar ionospheric-dispersion animation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .radar_physics import compressed_lfm_profile, plasma_group_velocity
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
    WHITE,
    Canvas,
    PlotBox,
    clamp,
    crossfade,
    ease,
    fade_color,
)


DURATION = 27.0


def _node(canvas: Canvas, box: tuple[int, int, int, int], title: str, detail: str, accent: tuple[int, int, int]) -> None:
    canvas.rect(box, PANEL, accent)
    canvas.text((box[0] + 24, box[1] + 34), title, 27, TEXT, bold=True)
    canvas.text((box[0] + 24, box[1] + 76), detail, 19, MUTED)


def render_intro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.2)
    canvas.text((110, 86), "场景二 · 宽带雷达", 24, fade_color(CYAN, reveal), bold=True)
    canvas.text((960, 390), "电离层色散如何模糊距离像", 58, fade_color(TEXT, reveal), bold=True, anchor="mm")
    canvas.text((960, 478), "宽带 LFM 回波 · 频率相关群时延 · 脉冲压缩失配", 29, fade_color(MUTED, reveal), anchor="mm")
    line = clamp((local_time - 0.4) / 1.1)
    canvas.line([(520, 558), (520 + 880 * line, 558)], fade_color(AMBER, line), 4)
    canvas.text((960, 650), "同一个回波，为何不再压缩成清晰尖峰？", 29, fade_color(TEXT, line), anchor="mm")
    return canvas.image


def render_link(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("01  真实系统位置", "星载宽带雷达链路", "色散发生在电离层传播段，后果在接收机脉冲压缩后显现")
    boxes = (
        (110, 310, 420, 430, "LFM 发射机", "产生宽带线性调频脉冲", CYAN),
        (555, 310, 895, 430, "电离层", "等离子体色散介质", VIOLET),
        (1030, 310, 1335, 430, "地面目标", "反射宽带回波", AMBER),
        (1470, 310, 1810, 430, "接收与压缩", "形成一维距离像", CORAL),
    )
    for left, top, right, bottom, title, detail, accent in boxes:
        _node(canvas, (left, top, right, bottom), title, detail, accent)
    for start, end in (((420, 370), (555, 370)), ((895, 370), (1030, 370)), ((1335, 370), (1470, 370))):
        canvas.arrow(start, end, GRID, 4)

    progress = (max(local_time, 0.0) / 3.6) % 1.0
    pulse_x = 430 + 1025 * progress
    canvas.circle((pulse_x, 370), 10, AMBER)
    canvas.text((960, 485), "往返传播两次穿越电离层，群时延误差累积", 24, AMBER, bold=True, anchor="mm")

    plot = PlotBox(170, 610, 1750, 920)
    canvas.plot_axes(plot)
    t = np.linspace(0.0, 1.0, 800)
    frequency = 0.16 + 0.68 * t
    canvas.plot_curve(plot, t, frequency, CYAN, y_min=0.0, y_max=1.0, width=5)
    canvas.text((plot.left - 24, plot.top + 10), "频率", 21, MUTED, anchor="ra")
    canvas.text((plot.right, plot.bottom + 28), "时间", 21, MUTED, anchor="ra")
    canvas.formula((plot.left + 45, plot.top + 22), "f(t) = f_{0} + Kt", 28, CYAN)
    canvas.text((960, 985), "发射时：频率与时间保持严格线性关系", 24, TEXT, anchor="mm")
    return canvas.image


def render_group_delay(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("02  电离层色散", "不同频率获得不同群时延", "高频分量更快，低频分量相对滞后；此处不引入幅度衰减")
    progress = ease(clamp(local_time / 6.8))
    canvas.formula((960, 285), "n(ω) = √(1 − ω_{p}^{2}/ω^{2})", 34, TEXT, anchor="mm")

    start_x, travel = 260, 1260
    for y, label, omega, color in ((455, "低频分量", 1.35, CORAL), (650, "高频分量", 2.7, CYAN)):
        canvas.line([(start_x, y), (start_x + travel, y)], GRID, 5)
        velocity = float(plasma_group_velocity(omega))
        position = start_x + travel * progress * velocity
        canvas.circle((position, y), 15, color)
        canvas.text((110, y), label, 24, color, bold=True, anchor="lm")
        canvas.formula((1550, y - 20), f"v_{{g}}/c = {velocity:.2f}", 24, color)

    canvas.line([(start_x, 790), (start_x + travel, 790)], GRID, 2)
    canvas.text((start_x, 830), "进入电离层", 20, MUTED, anchor="ma")
    canvas.text((start_x + travel, 830), "离开电离层", 20, MUTED, anchor="ma")
    lag = (float(plasma_group_velocity(2.7)) - float(plasma_group_velocity(1.35))) * progress
    canvas.formula((730, 900), "Δτ_{g} ∝ 1/f^{2}", 30, AMBER, anchor="mm")
    canvas.text((1015, 900), "相对时延", 24, MUTED, anchor="mm")
    canvas.formula((1235, 900), f"= {lag:.2f}", 30, AMBER, anchor="mm")
    canvas.text((960, 975), "线性调频信号原有的频率—时间对应关系被扭曲", 27, TEXT, bold=True, anchor="mm")
    return canvas.image


def render_compression(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("03  接收端后果", "脉冲压缩失配", "同一个匹配滤波器处理理想回波与色散回波")
    progress = ease(clamp(local_time / 6.0))
    delay, ideal = compressed_lfm_profile(0.0)
    _, dispersed = compressed_lfm_profile(60.0 * progress)
    mask = np.abs(delay) <= 0.19
    delay = delay[mask]
    ideal = ideal[mask]
    dispersed = dispersed[mask]

    left = PlotBox(125, 390, 900, 835)
    right = PlotBox(1020, 390, 1795, 835)
    for box, title, values, color in (
        (left, "无色散：匹配", ideal, CYAN),
        (right, "有色散：失配", dispersed, CORAL),
    ):
        canvas.text((box.left, box.top - 58), title, 28, color, bold=True)
        canvas.plot_axes(box)
        canvas.plot_curve(box, delay, values, color, y_min=0.0, y_max=1.08, width=5)
        canvas.text((box.right, box.bottom + 30), "时延", 21, MUTED, anchor="ra")

    canvas.text((512, 905), "窄主瓣 · 峰值集中", 25, CYAN, bold=True, anchor="mm")
    canvas.text((1408, 905), "主瓣展宽 · 峰值下降 · 旁瓣升高", 25, CORAL, bold=True, anchor="mm")
    canvas.formula((960, 987), "H(ω) = exp[−jφ(ω)]    |H(ω)| = 1", 29, TEXT, anchor="mm")
    return canvas.image


def render_range_image(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    canvas.header("04  工程表现", "距离像分辨能力下降", "相邻目标的回波主瓣变宽后开始相互覆盖")
    progress = ease(clamp(local_time / 3.5))
    x = np.linspace(-5.0, 5.0, 1000)
    narrow = np.exp(-np.square((x + 0.65) / 0.22)) + 0.82 * np.exp(-np.square((x - 0.65) / 0.22))
    width_factor = 0.22 + 0.68 * progress
    blurred = np.exp(-np.square((x + 0.65) / width_factor)) + 0.82 * np.exp(-np.square((x - 0.65) / width_factor))
    blurred /= np.max(blurred)
    box = PlotBox(140, 350, 1780, 820)
    canvas.plot_axes(box)
    canvas.plot_curve(box, x, narrow, fade_color(CYAN, 0.5), y_min=0.0, y_max=1.18, width=3)
    canvas.fill_curve(box, x, blurred, AMBER, y_min=0.0, y_max=1.18)
    canvas.plot_curve(box, x, blurred, AMBER, y_min=0.0, y_max=1.18, width=5)
    canvas.text((box.right, box.bottom + 30), "距离单元", 21, MUTED, anchor="ra")
    canvas.text((960, 900), "目标峰逐渐合并：定位偏差 · 图像模糊 · 虚警风险", 29, CORAL, bold=True, anchor="mm")
    canvas.formula((960, 982), "ΔR = cΔt/2", 32, TEXT, anchor="mm")
    return canvas.image


def render_outro(width: int, height: int, local_time: float) -> Image.Image:
    canvas = Canvas(width, height)
    reveal = ease(local_time / 1.0)
    canvas.text((960, 310), "电离层中的频率相关群时延", 31, fade_color(VIOLET, reveal), anchor="mm")
    canvas.arrow((960, 375), (960, 490), fade_color(MUTED, reveal), 4)
    canvas.text((960, 565), "LFM 相位关系被扭曲", 48, fade_color(AMBER, reveal), bold=True, anchor="mm")
    canvas.arrow((960, 630), (960, 745), fade_color(MUTED, reveal), 4)
    canvas.text((960, 820), "距离像展宽与模糊", 42, fade_color(CORAL, reveal), bold=True, anchor="mm")
    canvas.formula((960, 960), "v_{g} = dω/dk", 34, fade_color(TEXT, reveal), anchor="mm")
    return canvas.image


def render_frame(width: int, height: int, seconds: float) -> Image.Image:
    starts = (0.0, 2.2, 8.0, 15.2, 22.0, 25.2)
    renderers = (render_intro, render_link, render_group_delay, render_compression, render_range_image, render_outro)
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
