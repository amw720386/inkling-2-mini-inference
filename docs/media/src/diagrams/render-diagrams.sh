#!/usr/bin/env bash
# Re-render the architecture diagrams, the dspark / scheduler GIFs and the how-it-works scene.
#   docs/media/src/diagrams/render-diagrams.sh [out_dir]     (default: docs/media)
# Needs node 18+ (npm install in docs/media/src), ffmpeg, and headless Chrome
# (npx @puppeteer/browsers install chrome-headless-shell@stable, or set CHROME=/path).
set -euo pipefail
src=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${1:-$src/..}")
mkdir -p "$out"
cd "$src"
node diagrams/export-svg.js "$out"
gifs/gif.sh gifs/dspark.html "$out/dspark.gif" 960 540 20
gifs/gif.sh gifs/scheduler.html "$out/scheduler.gif" 960 540 20
node render.js scenes/howitworks.html "$out/scene-3-1x1.mp4" --w 1080 --h 1080
node render.js scenes/howitworks.html "$out/scene-3-16x9.mp4" --w 1920 --h 1080
