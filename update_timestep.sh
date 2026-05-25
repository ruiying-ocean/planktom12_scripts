#!/usr/bin/env bash
#
# update_timestep.sh — rescale the NEMO ocean timestep and every timestep-dependent
# counter consistently: the namelist counters AND setUpData's stepsPerYear.
#
# Changing the timestep by hand is painful: rn_Dt/rn_rdt plus nn_itend, nn_stock,
# nn_e and nn_fsbc must move together across the coldstart/restart/cycling
# namelists, and stepsPerYear (restart-step math) has to match. This does it in
# one shot. ALL scaling is by the dt ratio only, so it makes NO calendar
# (days-per-year / nn_leapy) assumption:
#
#   rn_Dt (NEMO5) | rn_rdt (NEMO3.6)   ->  new_dt
#   nn_itend, nn_stock, nn_fsbc        ->  * old_dt/new_dt   (steps per fixed duration)
#   nn_e   (barotropic sub-steps)      ->  * new_dt/old_dt   (keep sub-step length)
#   ln_1st_euler                       ->  .true.            (if present & not already)
#   setUpData stepsPerYear             ->  * old_dt/new_dt   (kept in sync)
#
# Example (NEMO5 TOM6 RK3, double to 3 h): rn_Dt 5400.->10800., nn_itend/nn_stock
# 5840->2920, nn_e 30->60, nn_fsbc 2->1, stepsPerYear 5840->2920.
#
# Target dt comes from setUpData (timestep:); --new-dt overrides it. Patches every
# *regular* namelist* file in DIR that defines the timestep (symlinks skipped —
# their targets are patched once), so point it at a run dir or an EXP00 template.
# Files already at the target dt are left untouched. All scalings are computed and
# checked to be clean integers BEFORE anything is written (no partial patches).
#
# Workflow: edit timestep: in setUpData, then run this (setUpRun runs it for you).
# Do not hand-edit stepsPerYear — this keeps it in step with the timestep.
#
# DRY-RUN by default; pass --apply to write. Uses GNU grep -P / sed
# --follow-symlinks / bash mapfile, so run it on the HPC. Review the dry-run first.

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 DIR [--new-dt SECONDS] [--apply]

  DIR          directory with the namelist files (run dir or EXP00 template)
  --new-dt N   target ocean timestep in seconds; default: timestep: from the
               setUpData in DIR
  --apply      write the changes (default: dry-run, just print what would change)

Examples:
  $0 ~/scratch/ModelRuns/TOM6_RK3                  # use setUpData timestep, preview
  $0 ~/scratch/ModelRuns/TOM6_RK3 --apply          # use setUpData timestep, write
  $0 /path/EXP00 --new-dt 10800 --apply            # explicit override, write
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

[ -n "$DIR" ] || { usage; exit 1; }
[ -d "$DIR" ] || { echo "Error: not a directory: $DIR" >&2; exit 1; }

# Default the target timestep to setUpData's timestep: (single source of truth).
SD=$(find "$DIR" -maxdepth 1 -type f -name 'setUpData*dat' 2>/dev/null | head -1 || true)
if [ -z "$NEW_DT" ]; then
    [ -n "$SD" ] && NEW_DT=$(grep -h '^timestep:' "$SD" 2>/dev/null | head -1 | cut -d':' -f2 || true)
    [ -n "$NEW_DT" ] || { echo "Error: no --new-dt given and no 'timestep:' in setUpData under $DIR" >&2; exit 1; }
    echo "Target timestep from $(basename "$SD"): ${NEW_DT}s"
fi
case "$NEW_DT" in (*[!0-9]*|"") echo "Error: timestep must be a positive integer (seconds): '$NEW_DT'" >&2; exit 1 ;; esac

# First numeric value of namelist key $1 in file $2 ("" if absent).
nl_get() { grep -oP '^\s*'"$1"'\s*=\s*\K[-0-9.]+' "$2" 2>/dev/null | head -1 || true; }

# Integer scaling with an exact-divisibility guard; die (exit 2) on non-exact so
# we never emit a fractional counter. Called during planning, before any write.
scale_exact() { # value, mul, div, label
    local v=$1 mul=$2 div=$3 label=$4
    if [ $(( (v * mul) % div )) -ne 0 ]; then
        echo "  ERROR: $label=$v does not scale to an integer ($v*$mul/$div); aborting (nothing written)." >&2
        exit 2
    fi
    echo $(( v * mul / div ))
}

# Target namelists: regular files named namelist* that define the timestep.
mapfile -t TARGETS < <(
    find "$DIR" -maxdepth 1 -type f -name 'namelist*' 2>/dev/null \
    | while read -r f; do grep -qP '^\s*(rn_Dt|rn_rdt)\s*=' "$f" 2>/dev/null && echo "$f"; done \
    | sort
)
[ ${#TARGETS[@]} -gt 0 ] || { echo "Error: no regular namelist* file defining rn_Dt/rn_rdt in $DIR" >&2; exit 1; }

echo "Target timestep: ${NEW_DT}s   mode: $([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN)"
echo

# --- Plan + validate everything first (no writes). PLAN entries: file<TAB>key<TAB>newval
PLAN=() ; SPY_PLAN="" ; run_old_dt=""
for f in "${TARGETS[@]}"; do
    if grep -qP '^\s*rn_Dt\s*=' "$f"; then dt_var=rn_Dt; else dt_var=rn_rdt; fi
    old_dt_raw=$(nl_get "$dt_var" "$f")
    old_dt=${old_dt_raw%.*}                 # strip trailing ".": 5400. -> 5400
    [ -n "$old_dt" ] || { echo "  skip (no $dt_var value): $(basename "$f")" >&2; continue; }
    run_old_dt=$old_dt                      # all phase namelists share one dt

    if [ "$old_dt" = "$NEW_DT" ]; then
        echo "== $(basename "$f"): already ${NEW_DT}s, unchanged"
        continue
    fi

    echo "== $(basename "$f")  ($dt_var ${old_dt} -> ${NEW_DT})"
    PLAN+=("$f"$'\t'"$dt_var"$'\t'"${NEW_DT}.")
    printf '   %-12s %s -> %s.\n' "$dt_var" "$old_dt_raw" "$NEW_DT"

    for k in nn_itend nn_stock nn_fsbc; do          # scale by old_dt/new_dt
        ov=$(nl_get "$k" "$f"); [ -n "$ov" ] || continue
        nv=$(scale_exact "$ov" "$old_dt" "$NEW_DT" "$k")
        PLAN+=("$f"$'\t'"$k"$'\t'"$nv"); printf '   %-12s %s -> %s\n' "$k" "$ov" "$nv"
    done
    ek=$(nl_get nn_e "$f")                            # scale by new_dt/old_dt
    if [ -n "$ek" ]; then
        ev=$(scale_exact "$ek" "$NEW_DT" "$old_dt" "nn_e")
        PLAN+=("$f"$'\t'"nn_e"$'\t'"$ev"); printf '   %-12s %s -> %s\n' "nn_e" "$ek" "$ev"
    fi
    le=$(grep -oP '^\s*ln_1st_euler\s*=\s*\K\.\w+\.' "$f" 2>/dev/null | head -1 || true)
    if [ -n "$le" ] && [ "$le" != ".true." ]; then
        PLAN+=("$f"$'\t'"ln_1st_euler"$'\t'".true."); printf '   %-12s %s -> .true.\n' "ln_1st_euler" "$le"
    fi
    echo
done

# stepsPerYear in setUpData, scaled by the SAME dt ratio (calendar-independent).
if [ -n "$SD" ] && [ -n "$run_old_dt" ] && [ "$run_old_dt" != "$NEW_DT" ]; then
    osp=$(grep -h '^stepsPerYear:' "$SD" | head -1 | cut -d':' -f2 || true)
    if [ -n "$osp" ]; then
        nsp=$(scale_exact "$osp" "$run_old_dt" "$NEW_DT" "stepsPerYear")
        SPY_PLAN="$SD"$'\t'"$nsp"
        echo "setUpData $(basename "$SD"): stepsPerYear ${osp} -> ${nsp}"
        echo
    fi
fi

if [ ${#PLAN[@]} -eq 0 ] && [ -z "$SPY_PLAN" ]; then
    echo "Nothing to do: already at ${NEW_DT}s."
    exit 0
fi

# --- Apply (all scalings already validated above).
if [ $APPLY -eq 1 ]; then
    for e in "${PLAN[@]}"; do
        IFS=$'\t' read -r file key val <<<"$e"
        if [ "$key" = "ln_1st_euler" ]; then
            sed --follow-symlinks -i -E "s/^([[:space:]]*ln_1st_euler[[:space:]]*=)[[:space:]]*\.[A-Za-z]+\./\1 .true./" "$file"
        else
            sed --follow-symlinks -i -E "s/^([[:space:]]*$key[[:space:]]*=)[[:space:]]*[-0-9.]+/\1 $val/" "$file"
        fi
    done
    if [ -n "$SPY_PLAN" ]; then
        IFS=$'\t' read -r file nsp <<<"$SPY_PLAN"
        sed -i -E "s/^(stepsPerYear:).*/\1${nsp}/" "$file"
    fi
    echo "Applied."
else
    echo "Dry-run only. Re-run with --apply to write."
fi
