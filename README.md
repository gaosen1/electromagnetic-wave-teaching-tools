# 系统色散动态演示

本工程包含三段独立教学动画：

1. 光纤通信：从双频复合波、群速度到脉冲展宽与码间干扰；
2. 宽带雷达：电离层色散导致 LFM 脉冲压缩失配与距离像模糊；
3. 高速数字电路：224 Gb/s PAM-4 信号经过 PCB 通道后的波形畸变与眼图闭合。

第一段把教材中的双频复合波扩展为一段 36 秒教学动画：

1. 两个相近频率的平面波叠加形成缓变包络；
2. 载波波峰以相速度移动，包络以群速度移动；
3. 有限带宽高斯脉冲在二阶色散下展宽；
4. OOK 序列中，展宽脉冲的尾部进入相邻码元采样点，抬高“0”码元并引发误判。

## 物理约定

- 教材复合波场景绘制电场 `E`；
- 光纤脉冲场景绘制光强 `I = |A|²`；
- 有色散场景采用二阶展开 `β(ω₀+Ω) ≈ β₀ + β₁Ω + β₂Ω²/2`；
- 展宽过程中保持 `∫I dt` 不变，不把色散画成损耗；
- 码间干扰场景使用 OOK 码元序列和固定判决门限；显示中的幅度已经接收端增益归一化，用于观察采样裕量；
- 线性色散只改变频谱相位，不绘制错误的幅度频谱展宽。
- 公式统一使用 STIX 数学字体，变量、希腊字母、运算符和数字按数学排版绘制；上下标使用独立字号和基线。
- 雷达场景使用冷、无碰撞、无磁等离子体的归一化群速度模型；匹配滤波劣化由纯相位通道产生。
- PCB 场景把频率相关群延迟与高频损耗分开显示，避免把色散和衰减混为一谈。
- PAM-4 发射眼图使用有限带宽的高斯低通发射模型，并标出三条判决门限和上、中、下三个眼开口。

## 一键构建

生成三段 1080p MP4，以及各自的 GIF 预览和封面图：

```bash
sh build_outputs.sh
```

## 单独渲染

依赖 Python 3、NumPy、Pillow 和 FFmpeg。当前工作区使用 Codex 随附的 Python 运行时：

```bash
/Users/gaosen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 render_animation.py
```

渲染单帧：

```bash
/Users/gaosen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 render_animation.py --demo radar --frame 17
```

运行物理测试：

```bash
/Users/gaosen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
```
