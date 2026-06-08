#!/bin/bash
# Compress NetCDF output in a local model run dir, in place.
# Usage: ./compress_nc.sh <model_id>
#
# Walks ~/scratch/ModelRuns/<model_id>/ for *.nc files and deflates each one
# with nccopy -d 4 -s, writing to a temp file first and then atomically
# swapping it into place. Two layouts are handled:
#   - symlinks into the AFM archive (/gpfs/afm/.../software/runs/...): the
#     target is compressed and the swap happens on AFM; the symlink is unchanged.
#   - real local files (e.g. fresh NEMO output not yet archived): compressed
#     in place on the run-dir filesystem.
# Already-deflated files are skipped, so this script is safe to re-run.

set -u

if [ $# -lt 1 ]; then
	echo "Usage: $0 <model_id>"
	exit 1
fi

model_id=$1
local_dir="$HOME/scratch/ModelRuns/$model_id"

if [ ! -d "$local_dir" ]; then
	echo "not a directory: $local_dir"
	exit 1
fi

cd "$local_dir" || exit 1

n_done=0
n_skip=0
n_fail=0
total_before=0
total_after=0

shopt -s nullglob
for link in *.nc; do
	case "$link" in
		*restart*) continue ;;
		*.tmp.nc) continue ;;   # leftover temp from an interrupted run
	esac
	if [ -L "$link" ]; then
		target=$(readlink -f "$link")
		if [ ! -f "$target" ]; then
			echo "skip (broken symlink): $link -> $target"
			continue
		fi
		case "$target" in
			*/software/runs/*) ;;
			*) echo "skip (not in archive): $link -> $target"; continue ;;
		esac
	elif [ -f "$link" ]; then
		target="$link"   # real local file: compress in place
	else
		echo "skip (neither file nor symlink): $link"
		continue
	fi

	if ncdump -hs "$target" 2>/dev/null | grep -q '_DeflateLevel = [1-9]'; then
		echo "skip (already deflated): $link"
		n_skip=$((n_skip + 1))
		continue
	fi

	local_tmp="${link%.nc}.tmp.nc"
	before=$(stat -c %s "$target")

	# 1) Compress source -> local tmp (fast local write)
	if ! nccopy -d 4 -s "$target" "$local_tmp"; then
		echo "FAILED nccopy: $link"
		rm -f "$local_tmp"
		n_fail=$((n_fail + 1))
		continue
	fi

	# Guard against a silently truncated/empty result before we destroy the original
	if [ ! -s "$local_tmp" ]; then
		echo "FAILED (empty output): $link"
		rm -f "$local_tmp"
		n_fail=$((n_fail + 1))
		continue
	fi

	# 2) Swap the compressed file into place
	if [ -L "$link" ]; then
		# Symlink target lives on AFM: stage next to the original, then atomic rename
		target_dir=$(dirname "$target")
		target_name=$(basename "$target")
		afm_tmp="$target_dir/.${target_name}.new"

		if ! cp "$local_tmp" "$afm_tmp"; then
			echo "FAILED cp to AFM: $link"
			rm -f "$local_tmp" "$afm_tmp"
			n_fail=$((n_fail + 1))
			continue
		fi

		if ! mv -f "$afm_tmp" "$target"; then
			echo "FAILED mv on AFM: $link"
			rm -f "$local_tmp" "$afm_tmp"
			n_fail=$((n_fail + 1))
			continue
		fi

		rm -f "$local_tmp"
	else
		# Real local file: tmp is on the same filesystem, rename over it directly
		if ! mv -f "$local_tmp" "$target"; then
			echo "FAILED mv: $link"
			rm -f "$local_tmp"
			n_fail=$((n_fail + 1))
			continue
		fi
	fi
	after=$(stat -c %s "$target")
	total_before=$((total_before + before))
	total_after=$((total_after + after))
	n_done=$((n_done + 1))
	echo "compressed: $link  ${before} -> ${after} bytes"
done

echo
echo "Done: $n_done compressed, $n_skip skipped, $n_fail failed"
if [ $n_done -gt 0 ]; then
	echo "Total: $total_before -> $total_after bytes"
fi
