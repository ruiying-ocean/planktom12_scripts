#!/bin/bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

fail() {
	echo "FAIL: $*" >&2
	exit 1
}

assert_file() {
	[ -f "$1" ] || fail "expected file: $1"
}

assert_symlink() {
	[ -L "$1" ] || fail "expected symlink: $1"
}

bash -n "$REPO_DIR"/workflow/{workflow_common,run_year,checkpoint_year,analyse_year,archive_year,report_run,submit_workflow,status}.sh \
	"$REPO_DIR/setUpRun.sh"

# Verify the submitted DAG without requiring a Slurm installation.
run_dir="$tmp_dir/run"
fake_bin="$tmp_dir/bin"
mkdir -p "$run_dir" "$fake_bin"
for script in workflow_common.sh run_year.sh checkpoint_year.sh analyse_year.sh \
	archive_year.sh report_run.sh submit_workflow.sh status.sh; do
	cp -p "$REPO_DIR/workflow/$script" "$run_dir/$script"
done

cat > "$run_dir/run.env" <<EOF
runId=TEST_RUN
yearStart=2000
yearEnd=2001
basedir=$tmp_dir/
modelDir=$run_dir
simulation=T001
Model=TEST
forcing_prefix=era
forcing_mode=transient
nemoVersion=NEMO36
executable=opa
iceRestartName=restart_ice_in
nemoCpus=2
xiosCpus=0
useXiosServer=false
timestep=5760
stepsPerYear=5475
spinupStart=1900
spinupEnd=1950
spinupRestartKeepFrequency=10
spinupOutputKeepFrequency=10
runRestartKeepFrequency=10
runOutputKeepFrequency=10
keepGrid_T=1
keepDiad=1
keepPtrc=1
keepIce=0
keepGrid_U=0
keepGrid_V=1
keepGrid_W=0
keepLimPhy=0
keepGflux=0
archiveDir=$tmp_dir/archive
outputFrequency=1m
toolkitDir=$run_dir/toolkit
MAMBA_EXE=/missing/mamba
MAMBA_ROOT_PREFIX=/missing
analysisEnv=base
EOF

cat > "$fake_bin/sbatch" <<'EOF'
#!/bin/bash
set -euo pipefail
counter=$(<"$SBATCH_COUNTER")
counter=$((counter + 1))
printf '%s\n' "$counter" > "$SBATCH_COUNTER"
printf '%s|' "$counter" >> "$SBATCH_LOG"
printf '%q ' "$@" >> "$SBATCH_LOG"
printf '\n' >> "$SBATCH_LOG"
printf '%s\n' "$counter"
EOF
cat > "$fake_bin/module" <<'EOF'
#!/bin/bash
exit 0
EOF
cat > "$fake_bin/ncdump" <<'EOF'
#!/bin/bash
exit 0
EOF
cat > "$fake_bin/nccopy" <<'EOF'
#!/bin/bash
set -euo pipefail
while [ "$#" -gt 2 ]; do shift; done
cp "$1" "$2"
EOF
chmod +x "$fake_bin/sbatch" "$fake_bin/module" "$fake_bin/ncdump" "$fake_bin/nccopy"
printf '0\n' > "$tmp_dir/counter"
: > "$tmp_dir/sbatch.log"

# Slurm relocates submitted scripts to /tmp/slurmd. Entry points must therefore
# find shared code through their explicit run-directory argument.
for script in run_year.sh checkpoint_year.sh analyse_year.sh archive_year.sh; do
	relocated="$tmp_dir/slurm-$script"
	cp -p "$run_dir/$script" "$relocated"
	if "$relocated" "$run_dir" >/dev/null 2>"$tmp_dir/relocated.err"; then
		fail "$script unexpectedly accepted a missing year argument"
	fi
	grep -q 'usage:' "$tmp_dir/relocated.err" || fail "$script could not load shared code after relocation"
done

PATH="$fake_bin:$PATH" SBATCH_COUNTER="$tmp_dir/counter" SBATCH_LOG="$tmp_dir/sbatch.log" \
	"$run_dir/submit_workflow.sh" "$run_dir" 2000 2001 compute >/dev/null

[ "$(wc -l < "$tmp_dir/sbatch.log" | tr -d ' ')" -eq 5 ] || fail "initial rolling slice should submit five jobs"
grep -q -- '--dependency=afterok:1' "$tmp_dir/sbatch.log" || fail "checkpoint must depend on first run"
grep -q $'^2001\tadvance\t5\t2$' "$run_dir/state/jobs.tsv" || fail "rolling advance job was not recorded"
grep -q 'submit_workflow.sh.*2001.*2001.*compute.*4' "$tmp_dir/sbatch.log" || fail "advance job did not carry rolling state"

# Simulate Slurm executing the small advance job after year 2000 checkpointing.
relocated_submit="$tmp_dir/slurm-submit_workflow.sh"
cp -p "$run_dir/submit_workflow.sh" "$relocated_submit"
PATH="$fake_bin:$PATH" SBATCH_COUNTER="$tmp_dir/counter" SBATCH_LOG="$tmp_dir/sbatch.log" \
	"$relocated_submit" "$run_dir" 2001 2001 compute 4 >/dev/null
[ "$(wc -l < "$tmp_dir/sbatch.log" | tr -d ' ')" -eq 10 ] || fail "second rolling slice should add five jobs"
grep -q -- '--dependency=afterok:7:4' "$tmp_dir/sbatch.log" || fail "analysis must wait for the prior archive"
grep -q $'^2001\treport\t10\t9$' "$run_dir/state/jobs.tsv" || fail "report job was not recorded"
"$run_dir/status.sh" "$run_dir" >/dev/null

# A partial range must stop after its final archive, without running the final report.
printf '0\n' > "$tmp_dir/counter"
: > "$tmp_dir/sbatch.log"
PATH="$fake_bin:$PATH" SBATCH_COUNTER="$tmp_dir/counter" SBATCH_LOG="$tmp_dir/sbatch.log" \
	"$run_dir/submit_workflow.sh" "$run_dir" 2000 2000 compute >/dev/null
[ "$(wc -l < "$tmp_dir/sbatch.log" | tr -d ' ')" -eq 4 ] || fail "partial range should submit four jobs"
if grep -q 'report_run.sh' "$tmp_dir/sbatch.log"; then
	fail "partial range unexpectedly submitted the final report"
fi

if PATH="$fake_bin:$PATH" SBATCH_COUNTER="$tmp_dir/counter" SBATCH_LOG="$tmp_dir/sbatch.log" \
	"$run_dir/submit_workflow.sh" "$run_dir" 2001 2001 compute >/dev/null 2>&1; then
	fail "continuation started without the prior checkpoint/archive markers"
fi

# Verify that checkpointing publishes every restart before the next run can start.
checkpoint_dir="$tmp_dir/checkpoint"
mkdir -p "$checkpoint_dir"
cp -p "$REPO_DIR/workflow/workflow_common.sh" "$REPO_DIR/workflow/checkpoint_year.sh" "$checkpoint_dir/"
sed \
	-e "s#runId=TEST_RUN#runId=CHECKPOINT_RUN#" \
	-e "s#modelDir=$run_dir#modelDir=$checkpoint_dir#" \
	-e "s#toolkitDir=$run_dir/toolkit#toolkitDir=$checkpoint_dir/toolkit#" \
	"$run_dir/run.env" > "$checkpoint_dir/run.env"

printf '5475\n' > "$checkpoint_dir/time.step"
: > "$checkpoint_dir/namelist_ref_other_years"
: > "$checkpoint_dir/namelist_ref"
: > "$checkpoint_dir/restart.nc"
: > "$checkpoint_dir/restart_ice_in.nc"
: > "$checkpoint_dir/restart_trc.nc"
: > "$checkpoint_dir/EMPave.dat"
: > "$checkpoint_dir/ocean.output"
: > "$checkpoint_dir/planktom-GR.log"
: > "$checkpoint_dir/planktom-ER.log"
for rank in 0000 0001; do
	for base in restart restart_ice restart_trc; do
		: > "$checkpoint_dir/ORCA2_00005475_${base}_${rank}.nc"
	done
done

relocated_checkpoint="$tmp_dir/slurm-checkpoint_year.sh"
cp -p "$checkpoint_dir/checkpoint_year.sh" "$relocated_checkpoint"
SLURM_JOB_ID=99 "$relocated_checkpoint" "$checkpoint_dir" 2000 >/dev/null

assert_file "$tmp_dir/archive/CHECKPOINT_RUN/ORCA2_00005475_restart_0000.nc"
assert_symlink "$checkpoint_dir/ORCA2_00005475_restart_0000.nc"
assert_symlink "$checkpoint_dir/restart_0000.nc"
assert_symlink "$checkpoint_dir/restart_ice_in_0001.nc"
assert_file "$checkpoint_dir/EMPave_2000.dat"
assert_file "$checkpoint_dir/state/2000/checkpoint.ok"
[ "$(<"$checkpoint_dir/state/2000/restart_step")" = 00005475 ] || fail "wrong restart timestep"

SLURM_JOB_ID=100 "$relocated_checkpoint" "$checkpoint_dir" 2000 >/dev/null
assert_file "$checkpoint_dir/state/2000/checkpoint.ok"

# NEMO5 checkpoint retries must reuse the recorded timestep and not advance the
# namelist twice if a scheduler or filesystem interruption causes a rerun.
nemo5_dir="$tmp_dir/nemo5-checkpoint"
mkdir -p "$nemo5_dir"
cp -p "$REPO_DIR/workflow/workflow_common.sh" "$REPO_DIR/workflow/checkpoint_year.sh" "$nemo5_dir/"
sed \
	-e 's/runId=TEST_RUN/runId=NEMO5_CHECKPOINT_RUN/' \
	-e "s#modelDir=$run_dir#modelDir=$nemo5_dir#" \
	-e 's/nemoVersion=NEMO36/nemoVersion=NEMO5/' \
	-e 's/executable=opa/executable=nemo/' \
	-e 's/iceRestartName=restart_ice_in/iceRestartName=restart_ice/' \
	-e "s#toolkitDir=$run_dir/toolkit#toolkitDir=$nemo5_dir/toolkit#" \
	"$run_dir/run.env" > "$nemo5_dir/run.env"

cat > "$nemo5_dir/namelist_cfg_other_years" <<'EOF'
&namrun
   nn_it000 = 1
   nn_itend = 5840
/
EOF
printf "cn_ocerst_out = 'restart_out'\n" > "$nemo5_dir/namelist_ref"
printf "cn_trcrst_out = 'restart_trc_out'\n" > "$nemo5_dir/namelist_top_ref"
printf "cn_icerst_out = 'restart_ice_out'\n" > "$nemo5_dir/namelist_ice_ref"
printf '5840\n' > "$nemo5_dir/time.step"
: > "$nemo5_dir/restart.nc"
: > "$nemo5_dir/restart_ice.nc"
: > "$nemo5_dir/restart_trc.nc"
for rank in 0000 0001; do
	for base in restart_out restart_ice_out restart_trc_out; do
		: > "$nemo5_dir/ORCA2_00005840_${base}_${rank}.nc"
	done
done

SLURM_JOB_ID=101 "$relocated_checkpoint" "$nemo5_dir" 2000 >/dev/null
grep -Eq 'nn_it000 *= *5841' "$nemo5_dir/namelist_cfg_other_years" || fail "wrong NEMO5 start step"
grep -Eq 'nn_itend *= *11680' "$nemo5_dir/namelist_cfg_other_years" || fail "wrong NEMO5 end step"
[ ! -e "$nemo5_dir/time.step" ] || fail "completed time.step was not rotated"
assert_symlink "$nemo5_dir/restart_0000.nc"
assert_symlink "$nemo5_dir/restart_ice_0001.nc"

SLURM_JOB_ID=102 "$relocated_checkpoint" "$nemo5_dir" 2000 >/dev/null
grep -Eq 'nn_it000 *= *5841' "$nemo5_dir/namelist_cfg_other_years" || fail "retry advanced NEMO5 start step twice"
grep -Eq 'nn_itend *= *11680' "$nemo5_dir/namelist_cfg_other_years" || fail "retry advanced NEMO5 end step twice"

# Verify archive publication and deletion policy with fake NetCDF commands.
archive_task_dir="$tmp_dir/archive-task"
mkdir -p "$archive_task_dir"
cp -p "$REPO_DIR/workflow/workflow_common.sh" "$REPO_DIR/workflow/archive_year.sh" "$archive_task_dir/"
sed \
	-e 's/runId=TEST_RUN/runId=ARCHIVE_TASK_RUN/' \
	-e "s#modelDir=$run_dir#modelDir=$archive_task_dir#" \
	-e "s#toolkitDir=$run_dir/toolkit#toolkitDir=$archive_task_dir/toolkit#" \
	"$run_dir/run.env" > "$archive_task_dir/run.env"
: > "$archive_task_dir/ORCA2_1m_20000101_20001231_grid_T.nc"
: > "$archive_task_dir/ORCA2_1m_20000101_20001231_grid_U.nc"

relocated_archive="$tmp_dir/slurm-archive_year.sh"
cp -p "$archive_task_dir/archive_year.sh" "$relocated_archive"
PATH="$fake_bin:$PATH" SLURM_JOB_ID=103 \
	"$relocated_archive" "$archive_task_dir" 2000 >/dev/null 2>&1
assert_file "$tmp_dir/archive/ARCHIVE_TASK_RUN/ORCA2_1m_20000101_20001231_grid_T.nc"
assert_symlink "$archive_task_dir/ORCA2_1m_20000101_20001231_grid_T.nc"
[ ! -e "$archive_task_dir/ORCA2_1m_20000101_20001231_grid_U.nc" ] || fail "discarded output was retained"
assert_file "$archive_task_dir/state/2000/archive.ok"

PATH="$fake_bin:$PATH" SLURM_JOB_ID=104 \
	"$relocated_archive" "$archive_task_dir" 2000 >/dev/null 2>&1
assert_symlink "$archive_task_dir/ORCA2_1m_20000101_20001231_grid_T.nc"

echo "Slurm workflow tests passed"
