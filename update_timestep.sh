#!/usr/bin/env bash
#
# update_timestep.sh — rescale the NEMO ocean timestep and all timestep-dependent
# namelist counters consistently, across every phase namelist in a directory.
#
# Changing the timestep by hand is painful: rn_Dt/rn_rdt plus nn_itend, nn_stock,
# nn_e and nn_fsbc all have to move together, in the coldstart / restart / cycling
# namelists. This does it in one shot, keeping the physical run length and the
# SBC/barotropic intervals constant.
#
# Scaling (r = new_dt / old_dt):
#   rn_Dt (NEMO5) | rn_rdt (NEMO3.6)  ->  new_dt
#   nn_itend, nn_stock, nn_fsbc       ->  old * old_dt/new_dt   (# steps for a fixed
#                                                                physical duration)
#   nn_e   (barotropic sub-steps)     ->  old * new_dt/old_dt   (keep sub-step length,
#                                                                e.g. ~180 s, constant)
#   ln_1st_euler                      ->  .true.                (required across a dt
#                                                                change; set if present)
# Example (NEMO5 TOM6, RK3, double the step to 3 h):
#   rn_Dt 5400.->10800.  nn_itend/nn_stock 5840->2920  nn_e 30->60  nn_fsbc 2->1
#
# It patches every *regular* file matching namelist* in DIR that defines the
# timestep (symlinks are skipped — their targets are patched once), so point it
# at a run dir (namelist_cfg_<forcing>_* copies) or an EXP00 template
# (namelist_ref*). If a setUpData*dat is present in DIR, stepsPerYear is updated
# to 31536000/new_dt as well.
#
# DRY-RUN by default; pass --apply to write. Run on the HPC where the namelists
# live. Always review the dry-run before --apply.

set -euo pipefail

SECONDS_PER_YEAR=31536000   # 365 d, no leap (NEMO 365_day calendar)

usage() {
    cat >&2 <<EOF
Usage: $0 DIR --new-dt SECONDS [--apply]

  DIR          directory containing the namelist files (run dir or EXP00 template)
  --new-dt N   target ocean timestep in seconds (e.g. 10800)
  --apply      write the changes (default: dry-run, just print what would change)

Example:
  $0 ~/scratch/ModelRuns/TOM6_RK3            --new-dt 10800            # preview
  $0 ~/scratch/ModelRuns/TOM6_RK3            --new-dt 10800 --apply    # write
EOF
}

DIR="" ; NEW_DT="" ; APPLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --new-dt) NEW_DT="$2"; shift 2 ;;
        --apply)  APPLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)  if [ -z "$DIR" ]; then DIR="$1"; shift; else echo "Unexpected arg: $1" >&2; exit 1; fi ;;
    esac
done

[ -n "$DIR" ] && [ -n "$NEW_DT" ] || { usage; exit 1; }
[ -d "$DIR" ] || { echo "Error: not a directory: $DIR" >&2; exit 1; }
case "$NEW_DT" in (*[!0-9]*|"") echo "Error: --new-dt must be a positive integer (seconds): '$NEW_DT'" >&2; exit 1 ;; esac

# Read the first numeric value of namelist key $1 in file $2 ("" if absent).
nl_get() { grep -oP '^\s*'"$1"'\s*=\s*\K[-0-9.]+' "$2" 2>/dev/null | head -1 || true; }

# Replace the numeric value of key $1 with $2 in file $3, preserving any trailing
# comment. --follow-symlinks is harmless on a regular file.
nl_set() {
    sed --follow-symlinks -i -E "s/^([[:space:]]*$1[[:space:]]*=)[[:space:]]*[-0-9.]+/\1 $2/" "$3"
}

# Integer scaling with an exact-divisibility guard. die on non-exact so we never
# emit a fractional step count.
scale_exact() { # value, mul, div, label, file
    local v=$1 mul=$2 div=$3 label=$4 file=$5
    if [ $(( (v * mul) % div )) -ne 0 ]; then
        echo "  ERROR: $label=$v does not scale to an integer ($v*$mul/$div); aborting." >&2
        echo "         (check the timestep ratio divides $label cleanly)" >&2
        exit 2
    fi
    echo $(( v * mul / div ))
}

# Find target namelists: regular files named namelist* that define the timestep.
mapfile -t TARGETS < <(
    find "$DIR" -maxdepth 1 -type f -name 'namelist*' 2>/dev/null \
    | while read -r f; do
        if grep -qP '^\s*(rn_Dt|rn_rdt)\s*=' "$f" 2>/dev/null; then echo "$f"; fi
      done | sort
)

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "Error: no regular namelist* file defining rn_Dt/rn_rdt found in $DIR" >&2
    exit 1
fi

echo "Target timestep: ${NEW_DT}s   mode: $([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN)"
echo "Namelists (${#TARGETS[@]}):"
printf '  %s\n' "${TARGETS[@]}"
echo

for f in "${TARGETS[@]}"; do
    # Which timestep variable does this file use?
    if grep -qP '^\s*rn_Dt\s*=' "$f"; then dt_var=rn_Dt; else dt_var=rn_rdt; fi
    old_dt_raw=$(nl_get "$dt_var" "$f")
    old_dt=${old_dt_raw%.*}          # strip trailing ".": 5400. -> 5400
    [ -n "$old_dt" ] || { echo "  skip (no $dt_var value): $f" >&2; continue; }

    echo "== $(basename "$f")  ($dt_var: ${old_dt} -> ${NEW_DT})"

    # Build the change set (only for keys present in this file).
    declare -a keys=() olds=() news=()
    keys+=("$dt_var"); olds+=("$old_dt_raw"); news+=("${NEW_DT}.")

    for k in nn_itend nn_stock nn_fsbc; do          # scale by old_dt/new_dt
        ov=$(nl_get "$k" "$f"); [ -n "$ov" ] || continue
        nv=$(scale_exact "$ov" "$old_dt" "$NEW_DT" "$k" "$f")
        keys+=("$k"); olds+=("$ov"); news+=("$nv")
    done
    ek=$(nl_get nn_e "$f")                            # scale by new_dt/old_dt
    if [ -n "$ek" ]; then
        ev=$(scale_exact "$ek" "$NEW_DT" "$old_dt" "nn_e" "$f")
        keys+=("nn_e"); olds+=("$ek"); news+=("$ev")
    fi

    for i in "${!keys[@]}"; do
        printf '   %-10s %s -> %s\n' "${keys[$i]}" "${olds[$i]}" "${news[$i]}"
    done
    le=$(grep -oP '^\s*ln_1st_euler\s*=\s*\K\.\w+\.' "$f" 2>/dev/null | head -1 || true)
    [ -n "$le" ] && printf '   %-10s %s -> .true.\n' "ln_1st_euler" "$le"

    if [ $APPLY -eq 1 ]; then
        for i in "${!keys[@]}"; do
            nl_set "${keys[$i]}" "${news[$i]}" "$f"
        done
        [ -n "$le" ] && sed --follow-symlinks -i -E \
            "s/^([[:space:]]*ln_1st_euler[[:space:]]*=)[[:space:]]*\.[A-Za-z]+\./\1 .true./" "$f"
    fi
    unset keys olds news
    echo
done

# Keep stepsPerYear (restart-step math) in sync with the new timestep.
sd=$(find "$DIR" -maxdepth 1 -type f -name 'setUpData*dat' 2>/dev/null | head -1 || true)
if [ -n "$sd" ]; then
    if [ $(( SECONDS_PER_YEAR % NEW_DT )) -ne 0 ]; then
        echo "WARNING: ${SECONDS_PER_YEAR}/${NEW_DT} is not an integer; stepsPerYear NOT updated in $(basename "$sd")" >&2
    else
        new_spy=$(( SECONDS_PER_YEAR / NEW_DT ))
        old_spy=$(grep -h '^stepsPerYear:' "$sd" | head -1 | cut -d':' -f2 || true)
        echo "setUpData $(basename "$sd"): stepsPerYear ${old_spy:-<unset>} -> ${new_spy}"
        if [ $APPLY -eq 1 ]; then
            if [ -n "$old_spy" ]; then
                sed -i -E "s/^(stepsPerYear:).*/\1${new_spy}/" "$sd"
            else
                printf 'stepsPerYear:%s\n' "$new_spy" >> "$sd"
            fi
        fi
    fi
fi

if [ $APPLY -eq 1 ]; then
    echo "Applied. Re-run NEMO from a coldstart or ensure ln_1st_euler handling on the dt change."
else
    echo "Dry-run only. Re-run with --apply to write."
fi
