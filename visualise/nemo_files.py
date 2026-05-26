"""Single source of truth for NEMO/ORCA2 output filenames.

Every visualise script builds the same ``ORCA2_1m_<year>0101_<year>1231_<kind>_T.nc``
name; centralising it here means the monthly-mean ORCA2 naming convention is
defined in one place. Stdlib-only so it can be imported from anywhere in the
pipeline (single-model and multimodel) without pulling in the plotting stack.
"""

from pathlib import Path


def nemo_file(run_dir, year, file_type):
    """Path to a NEMO ORCA2 monthly-mean output file for ``year``.

    ``file_type`` is the grid kind *including* its ``_T`` suffix, e.g. ``ptrc_T``,
    ``grid_T``, ``diad_T`` -- matching the on-disk name.
    """
    return Path(run_dir) / f"ORCA2_1m_{year}0101_{year}1231_{file_type}.nc"


def nemo_glob(run_dir, file_type):
    """Sorted list of NEMO ORCA2 monthly-mean files of ``file_type`` across all years."""
    return sorted(Path(run_dir).glob(f"ORCA2_1m_*_{file_type}.nc"))
