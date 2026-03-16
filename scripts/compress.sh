#!/usr/bin/env bash
# Re-encode a video with libx264 veryfast for sharing.
# Usage: ./scripts/compress.sh output/output.mp4
set -euo pipefail
input="$1"
output="${input%.mp4}_compressed.mp4"
ffmpeg -y -i "$input" -c:v libx264 -preset veryfast -pix_fmt yuv420p "$output"
echo "→ $output ($(du -h "$output" | cut -f1))"
