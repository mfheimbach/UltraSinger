#!/usr/bin/env bash
# Fetch a small Creative Commons dev set (with audio) from UltraStar-Deluxe/songs
# into ./eval_data/dev, one folder per song, ready for run_eval.py.
#   bash eval/fetch_dev_set.sh [target_dir]
set -euo pipefail

TARGET="${1:-eval_data/dev}"
SRC="${TARGET}/.src"
SONGS=(
  "Jonathan Coulton - Code Monkey"
  "Systemabsturz - Verdächtig"
  "Shearer - Stay with me"
  "Silver Note - Chasing Marks"
  "Steven Dunston - Northern Star"
  "Dead Smiling Pirates - I"
  "Fairy Bot Orchestra - Heaven can't wait"
)

if [ ! -d "$SRC/.git" ]; then
  git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/UltraStar-Deluxe/songs.git "$SRC"
fi
patterns=()
for s in "${SONGS[@]}"; do
  patterns+=("/Creative Commons/$s/song.txt" "/Creative Commons/$s/audio.*" "/Creative Commons/$s/license.txt")
done
git -C "$SRC" sparse-checkout init --no-cone
git -C "$SRC" sparse-checkout set "${patterns[@]}"
git -C "$SRC" checkout -q HEAD

for s in "${SONGS[@]}"; do
  src="$SRC/Creative Commons/$s"
  dst="$TARGET/$s"
  mkdir -p "$dst"
  audio=$(ls "$src"/audio.* | head -1)
  cp "$audio" "$dst/"
  cp "$src/license.txt" "$dst/"
  # point #MP3/#AUDIO at the file that actually exists (some upstream txts are stale)
  sed -E "s/^#(MP3|AUDIO):.*/#\1:$(basename "$audio")/" "$src/song.txt" > "$dst/song.txt"
done
echo "dev set ready in $TARGET (${#SONGS[@]} songs)"
