#!/usr/bin/env bash
# Render a GIF scene frame by frame and encode it as a looping GIF with one shared palette.
#   gifs/gif.sh <scene.html> <out.gif> [width] [height] [fps]
# Run from docs/media/src. Needs node (npm install), headless Chrome and ffmpeg.
set -euo pipefail
scene=$1 out=$2 w=${3:-960} h=${4:-540} fps=${5:-20}
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
node render.js "$scene" "$tmp/scene.mp4" --w "$w" --h "$h" --fps "$fps" --crf 4
ffmpeg -loglevel error -y -i "$tmp/scene.mp4" -vf "palettegen=max_colors=128:stats_mode=full" "$tmp/pal.png"
ffmpeg -loglevel error -y -i "$tmp/scene.mp4" -i "$tmp/pal.png" \
  -lavfi "paletteuse=dither=sierra2_4a:diff_mode=rectangle" -loop 0 "$out"
echo "$out: $(du -k "$out" | cut -f1) KiB"
