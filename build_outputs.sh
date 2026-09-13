#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/Users/gaosen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}
build_demo() {
  demo=$1
  stem=$2
  poster_time=$3
  video="$PROJECT_DIR/output/$stem.mp4"
  gif="$PROJECT_DIR/output/${stem}_预览.gif"
  poster="$PROJECT_DIR/output/${stem}_封面.png"

  "$PYTHON_BIN" "$PROJECT_DIR/render_animation.py" --demo "$demo" --output "$video"
  ffmpeg -hide_banner -loglevel error -y -i "$video" \
    -vf "fps=12,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
    "$gif"
  ffmpeg -hide_banner -loglevel error -y -ss "$poster_time" -i "$video" -frames:v 1 "$poster"
  printf '%s\n' "$video" "$gif" "$poster"
}

build_demo fiber "系统色散动态演示_需求一" 20
build_demo radar "系统色散动态演示_场景二_宽带雷达" 17
build_demo pcb "系统色散动态演示_场景三_高速PCB" 20
