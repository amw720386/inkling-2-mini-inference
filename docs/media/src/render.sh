#!/usr/bin/env bash
# Re-render every video, GIF and diagram in docs/media from the sources in docs/media/src.
#
#   docs/media/src/render.sh            # everything
#   docs/media/src/render.sh video      # hero video only (both aspect ratios)
#   docs/media/src/render.sh gifs       # README GIFs only
#
# Needs: node >= 18, ffmpeg, and Chrome or chrome-headless-shell (set CHROME=/path if render.js cannot find it).
# Intermediate scene clips go to $WORK (default: a temp dir) and are deleted at the end.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="$(cd "$SRC/.." && pwd)"
if [ -z "${WORK:-}" ]; then WORK=$(mktemp -d); OWN_WORK=1; else mkdir -p "$WORK"; OWN_WORK=0; fi
WHAT="${1:-all}"
XF=0.4   # cross-fade between scenes, seconds

[ -d "$SRC/node_modules" ] || (cd "$SRC" && npm ci --no-audit --no-fund)

# Scenes of the hero video, in order. Each page follows the render.js contract.
SCENES=(scene1-opener scene2-race scene3-how scene4-memory endcard)

render_scene() {  # name WxH tag
  local w=${2%x*} h=${2#*x}
  node "$SRC/render.js" "$SRC/scenes/$1.html" "$WORK/$1-$3.mp4" --w "$w" --h "$h" --fps 30 --crf 12
}

clip_dur() {  # seconds, from ffmpeg's header line (no ffprobe needed)
  ffmpeg -hide_banner -i "$1" 2>&1 | sed -n 's/.*Duration: \([0-9:.]*\).*/\1/p' | awk -F: '{print $1*3600 + $2*60 + $3}'
}

# Join clips with $XF s cross-fades (each clip keeps its first and last $XF s visually stable).
join_clips() {  # out clip...
  local out=$1; shift
  local args=() filt="" prev="[0:v]" off=0 i=0
  for c in "$@"; do args+=(-i "$c"); done
  for c in "$@"; do
    local d; d=$(clip_dur "$c")
    if [ $i -gt 0 ]; then
      filt+="${prev}[$i:v]xfade=transition=fade:duration=$XF:offset=$off[v$i];"
      prev="[v$i]"
    fi
    off=$(python3 -c "print(round($off + $d - $XF, 3))")
    i=$((i + 1))
  done
  ffmpeg -y -loglevel error "${args[@]}" -filter_complex "${filt%;}" -map "$prev" \
    -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart "$out"
}

# GIF from a clip: 2-pass palette, width $2, fps $3.
to_gif() {  # in.mp4 width fps out.gif
  ffmpeg -y -loglevel error -i "$1" -vf "fps=$3,scale=$2:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=128" "$WORK/pal.png"
  ffmpeg -y -loglevel error -i "$1" -i "$WORK/pal.png" \
    -lavfi "fps=$3,scale=$2:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$4"
}

if [ "$WHAT" = all ] || [ "$WHAT" = video ]; then
  for size in 1080x1080:1x1 1920x1080:16x9; do
    clips=()
    for s in "${SCENES[@]}"; do
      render_scene "$s" "${size%:*}" "${size#*:}"
      clips+=("$WORK/$s-${size#*:}.mp4")
    done
    join_clips "$OUT/inkling-launch-${size#*:}.mp4" "${clips[@]}"
    echo "wrote $OUT/inkling-launch-${size#*:}.mp4"
  done
fi

if [ "$WHAT" = all ] || [ "$WHAT" = gifs ]; then
  for g in race loop memory dspark scheduler; do
    [ -f "$SRC/scenes/gif-$g.html" ] || { echo "skip $g.gif (no scenes/gif-$g.html)"; continue; }
    node "$SRC/render.js" "$SRC/scenes/gif-$g.html" "$WORK/gif-$g.mp4" --w 960 --h 540 --fps 30 --crf 12
    to_gif "$WORK/gif-$g.mp4" 960 15 "$OUT/$g.gif"
    echo "wrote $OUT/$g.gif ($(du -h "$OUT/$g.gif" | cut -f1))"
  done
fi

[ "$OWN_WORK" = 1 ] && rm -rf -- "${WORK:?}"
exit 0
