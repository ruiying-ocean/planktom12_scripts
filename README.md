# PlankTOM workflow toolkit

This repository prepares, runs, monitors, analyses, and archives NEMO–PlankTOM
experiments on Slurm.

Slurm owns workflow scheduling. Bash owns model staging, restart publication,
and archival. Python is used only for NetCDF analysis and visualisation.

Repository responsibilities are separated into `workflow/` (Slurm and shell
operations), `analyser/` (scientific reduction), `visualise/` (plots and
reports), and `configs/` (run inputs).

## Workflow

Each model year is represented by explicit Slurm jobs:

```text
run[Y] -> checkpoint[Y] -> run[Y+1]
                 |
                 +-> analyse[Y] -> archive[Y] -> analyse[Y+1]

archive[final year] -> report
```

- `run_year.sh` runs NEMO and writes no scheduler state.
- `checkpoint_year.sh` validates every restart rank, advances the active
  namelist, publishes restart files to AFM, and records the timestep.
- `analyse_year.sh` computes AMOC and analyser statistics, then refreshes the
  monitoring time series.
- `archive_year.sh` compresses and publishes the configured output types.
- `report_run.sh` applies retention policy and creates the final plots and HTML
  report.
- `submit_workflow.sh` submits these tasks with native Slurm `afterok`
  dependencies.

The model chain waits only for checkpoints. The post-processing chain is
serialized separately, preventing concurrent writes to analyser CSV files
without delaying the next model year.

## Starting a run

```bash
./setUpRun.sh configs/setUpData_TOM6_JRA.dat TOM6_RY_EX01
```

An optional third argument initializes the run from another experiment's
spin-up restart:

```bash
./setUpRun.sh configs/setUpData_TOM6_JRA.dat TOM6_RY_EX01 TOM6_RY_SPIN
```

`setUpRun.sh`:

1. creates and validates the model directory;
2. stages inputs, executable, XML, and namelists;
3. copies the selected analyser and visualisation configuration;
4. creates a detached snapshot of the repository's analysis code under
   `.planktom_toolkit/`;
5. writes the resolved, shell-quoted `run.env`;
6. copies the workflow scripts into the run directory; and
7. submits the configured year range.

The code snapshot is deliberate: later edits to this repository do not change
the analysis or report implementation attached to an existing run.

## Configuration

Existing `configs/setUpData_*.dat` files remain the user-facing configuration.
They contain `name:value` records for:

- model version, executable, ranks, and years;
- forcing and restart inputs;
- namelist and XML sources;
- output and restart retention frequencies; and
- analyser and visualisation configurations.

Paths may contain additional `:` characters; setup splits each record only at
the first colon.

The archive root defaults to:

```text
/gpfs/afm/greenocean/software/runs
```

Set `archiveDir:` in a setup file to override it. `run.env` is generated output,
not a configuration file to maintain by hand.

## Partition selection

Setup submits with partition mode `auto`: it selects `compute` when the user
already has at least two jobs in `ib`, otherwise `ib`.

To submit a selected range or force a partition from an already prepared run:

```bash
cd ~/scratch/ModelRuns/TOM6_RY_EX01
./submit_workflow.sh "$PWD" 2000 2010 compute
```

The final report job is submitted only when the selected range reaches the
configured `yearEnd`. A range beginning after `yearStart` is accepted only when
the preceding year's checkpoint and archive success markers exist, preventing
an accidental continuation from stale restart state.

## Monitoring

From any prepared run:

```bash
./status.sh "$PWD"
```

This displays:

- the recorded task graph from `state/jobs.tsv`;
- active Slurm state from `squeue`;
- historical Slurm state from `sacct`; and
- durable success markers under `state/<year>/<task>.ok`.

Task logs are stored under `logs/<year>/`. The active NEMO stdout and stderr
remain `planktom-GR.log` and `planktom-ER.log`; checkpointing preserves a copy
for each year.

Slurm dependencies use `afterok`. A failed model year therefore prevents its
checkpoint and later model years from starting. A failed analysis blocks later
post-processing, while the model chain can continue from already validated
checkpoints.

## Run directory

```text
<run-id>/
├── run.env
├── setUpData_*.dat
├── analyser_config.toml
├── visualise_config.toml
├── .planktom_toolkit/
│   ├── analyser/
│   ├── shared/
│   └── visualise/
├── state/
│   ├── jobs.tsv
│   └── <year>/
│       ├── restart_step
│       └── *.ok
├── logs/<year>/
├── MOC/
└── ORCA2_*.nc
```

Retained output and restart files in the run directory become symlinks to the
AFM archive only after a complete copy has been published.

## Analysis and reports

Annual analysis is normally submitted by `submit_workflow.sh`. A prepared task
can also be submitted manually when repairing one year:

```bash
sbatch --partition=compute \
  ~/scratch/ModelRuns/TOM6_RY_EX01/analyse_year.sh \
  ~/scratch/ModelRuns/TOM6_RY_EX01 2000
```

Single-model monitoring output is written to:

```text
~/scratch/ModelRuns/monitor/<run-id>/
```

Multi-model comparison tools remain under `visualise/multimodel/` and consume
the analyser files produced by this workflow.

## Requirements

- Slurm (`sbatch`, `squeue`, and `sacct`)
- NEMO and optional XIOS executables
- the site MPI, NetCDF, HDF5, Ferret, and CDFtools installations
- the configured Mamba environment with NumPy, Xarray, pandas, Matplotlib,
  Cartopy, netCDF4, SciPy, GSW, and related analysis packages
- Quarto for HTML reports

## Checks

The shell workflow has a self-contained test using a fake `sbatch` command and
temporary restart files:

```bash
test/test_slurm_workflow.sh
```

It verifies shell syntax, dependency ordering, serialized post-processing,
partial-range safety, restart publication, and idempotent NEMO5 checkpoint
retries without requiring access to a Slurm cluster.

## License

MIT. See `LICENSE`.
