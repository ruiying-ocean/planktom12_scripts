#!/usr/bin/env bash
set -euo pipefail

AFM_ROOT="/gpfs/afm/greenocean/software/runs"
SCRATCH_ROOT="$HOME/scratch/ModelRuns"
# STEP_PER_YEAR (NEMO timesteps/year, for restart-step math) is read per-run
# from the run's setUpData below -- it depends on NEMO version / rn_rdt.

usage() {
    echo "Usage: $0 [--follow-symlink] [--afm] RUN_NAME [MODE]"
    echo "  MODE: fixed    - keep latest, -20yr, -40yr (default)"
    echo "        adaptive - keep first, middle, latest"
    echo "  --follow-symlink: also delete the target file when a restart is a symlink"
    echo "  --afm: operate on the AFM archive dir ($AFM_ROOT/RUN_NAME);"
    echo "         also removes matching scratch symlinks pointing at deleted files"
}

FOLLOW_SYMLINK=0
AFM_MODE=0
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --follow-symlink)
            FOLLOW_SYMLINK=1
            shift
            ;;
        --afm)
            AFM_MODE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage >&2
    exit 1
fi

RUN_NAME=$1
MODE=${2:-fixed}

if [[ "$MODE" != "fixed" && "$MODE" != "adaptive" ]]; then
    echo "Error: MODE must be 'fixed' or 'adaptive', got '$MODE'"
    exit 1
fi

if [[ "$AFM_MODE" == 1 ]]; then
    DIR="${AFM_ROOT}/${RUN_NAME}"
    SCRATCH_DIR="${SCRATCH_ROOT}/${RUN_NAME}"
else
    DIR="${SCRATCH_ROOT}/${RUN_NAME}"
    SCRATCH_DIR=""
fi

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR not found" >&2
    exit 1
fi

# Resolve steps-per-year from the run's setUpData (run dir, then scratch).
# No version assumption when set; version-aware fallback only if absent.
dat=""
for d in "$DIR" "$SCRATCH_DIR"; do
    [ -n "$d" ] || continue
    for c in "$d"/setUpData*dat; do
        [ -f "$c" ] && { dat="$c"; break; }
    done
    [ -n "$dat" ] && break
done
nemoVersion=""
STEP_PER_YEAR=""
if [ -n "$dat" ]; then
    nemoVersion=$(grep "^nemoVersion:" "$dat" | head -1 | cut -d':' -f2 || true)
    STEP_PER_YEAR=$(grep "^stepsPerYear:" "$dat" | head -1 | cut -d':' -f2 || true)
fi
if [ -z "$STEP_PER_YEAR" ]; then
    if [ "$nemoVersion" = "NEMO5" ]; then STEP_PER_YEAR=5840; else STEP_PER_YEAR=5475; fi
    echo "Note: stepsPerYear not in setUpData; using ${STEP_PER_YEAR} (nemoVersion='${nemoVersion:-unknown}')" >&2
fi

shopt -s nullglob

for family in "restart_" "restart_ice_" "restart_trc_"; do
    files=( "$DIR"/ORCA2_*_${family}[0-9][0-9][0-9][0-9].nc )
    [[ -e "${files[0]}" ]] || continue

    # find first and latest steps
    sorted=($(printf '%s\n' "${files[@]}" | sort -V))
    first="${sorted[0]}"
    latest="${sorted[-1]}"

    first_step_str=$(echo "$first" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
    first_step=$((10#$first_step_str))

    latest_step_str=$(echo "$latest" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
    latest_step=$((10#$latest_step_str))

    if [[ "$MODE" == "adaptive" ]]; then
	middle_step=$(( (first_step + latest_step) / 2 ))
	# round middle to nearest year boundary
	middle_step=$(( (middle_step / STEP_PER_YEAR) * STEP_PER_YEAR ))
	keep_steps=(
	    $first_step
	    $middle_step
	    $latest_step
	)
    else
	# fixed mode: latest, -20yr, -40yr
	keep_steps=(
	    $latest_step
	    $(( latest_step - 20*STEP_PER_YEAR ))
	    $(( latest_step - 40*STEP_PER_YEAR ))
	)
    fi

    echo "Run ${RUN_NAME} | family=${family%_} | keep steps: ${keep_steps[*]}"

    declare -A keep_map=()
    for k in "${keep_steps[@]}"; do ((k>=0)) && keep_map["$k"]=1; done

    for f in "${files[@]}"; do
	step_str=$(echo "$f" | sed -n 's/.*_\([0-9]\{8\}\)_'${family}'.*/\1/p')
	step=$((10#$step_str))
	if [[ -z "${keep_map[$step]+x}" ]]; then
	    if [[ "$FOLLOW_SYMLINK" == 1 && -L "$f" ]]; then
		target=$(readlink -f "$f")
		[[ -n "$target" && -e "$target" ]] && rm -f "$target"
	    fi
	    if [[ "$AFM_MODE" == 1 && -n "$SCRATCH_DIR" ]]; then
		# Remove matching scratch symlink so it doesn't dangle.
		scratch_link="${SCRATCH_DIR}/$(basename "$f")"
		if [ -L "$scratch_link" ]; then
		    link_target=$(readlink -f "$scratch_link" 2>/dev/null || true)
		    afm_target=$(readlink -f "$f" 2>/dev/null || true)
		    if [[ -n "$link_target" && "$link_target" == "$afm_target" ]]; then
			rm -f "$scratch_link"
		    fi
		fi
	    fi
	    rm "$f"
	fi
    done
done
