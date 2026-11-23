#!/usr/bin/env bash
# Re-render every video, GIF and diagram in docs/media from the sources in docs/media/src.
#
#   docs/media/src/render.sh            # everything
#   docs/media/src/render.sh video      # hero video only (both aspect ratios)
#   docs/media/src/render.sh gifs       # README GIFs only
#   docs/media/src/render.sh diagrams   # SVG diagrams only
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
SCENES=(3d/opener.html scenes/scene2-race.html scenes/howitworks.html 3d/memory.html scenes/endcard.html)

render_scene() {  # page WxH out.mp4
  local w=${2%x*} h=${2#*x}
  node "$SRC/render.js" "$SRC/$1" "$3" --w "$w" --h "$h" --fps 30 --crf 12
}

clip_dur() {  # seconds, from ffmpeg's header line (no ffprobe needed)
  { ffmpeg -hide_banner -i "$1" 2>&1 || true; } | sed -n 's/.*Duration: \([0-9:.]*\).*/\1/p' | awk -F: '{print $1*3600 + $2*60 + $3}'
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

if [ "$WHAT" = all ] || [ "$WHAT" = video ]; then
  for size in 1080x1080:1x1 1920x1080:16x9; do
    clips=(); n=0
    for s in "${SCENES[@]}"; do
      n=$((n + 1))
      render_scene "$s" "${size%:*}" "$WORK/scene$n-${size#*:}.mp4"
      clips+=("$WORK/scene$n-${size#*:}.mp4")
    done
    join_clips "$OUT/inkling-launch-${size#*:}.mp4" "${clips[@]}"
    echo "wrote $OUT/inkling-launch-${size#*:}.mp4"
  done
fi

if [ "$WHAT" = all ] || [ "$WHAT" = gifs ]; then
  # page  render size  fps  out  dither   (render larger than 960 where labels need it; every GIF is scaled to 960 px wide;
  # flat 2D scenes use no dither, which keeps card fills clean; the 3D scenes need bayer for their gradients)
  while read -r page size fps name dither; do
    node "$SRC/render.js" "$SRC/$page" "$WORK/$name.mp4" --w "${size%x*}" --h "${size#*x}" --fps "$fps" --crf 4
    ffmpeg -y -loglevel error -i "$WORK/$name.mp4" -vf "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=${dither}:diff_mode=rectangle" -loop 0 "$OUT/$name.gif"
    echo "wrote $OUT/$name.gif ($(du -h "$OUT/$name.gif" | cut -f1))"
  done <<'GIFS'
scenes/gif-race.html 960x540 15 race none
3d/opener.html?gif=1 1280x720 15 loop bayer:bayer_scale=4
3d/memory.html?gif=1 1280x720 15 memory bayer:bayer_scale=4
gifs/dspark.html 960x540 20 dspark none
GIFS
fi

if [ "$WHAT" = all ] || [ "$WHAT" = diagrams ]; then
  (cd "$SRC" && node diagrams/export-svg.js "$OUT")
fi

[ "$OWN_WORK" = 1 ] && rm -rf -- "${WORK:?}"
exit 0
