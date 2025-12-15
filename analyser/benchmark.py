#!/usr/bin/env python
"""
Benchmark script for analyser performance testing.

This script measures the performance of different components of the analyser
system to help identify bottlenecks and measure improvements.

Usage:
    python benchmark.py [config_file] [year_from] [year_to]

Example:
    python benchmark.py analyser_config.toml 2000 2005
"""

import sys
import time
import argparse
import tracemalloc
from pathlib import Path
import numpy as np

# Timing decorator
def timed(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


def benchmark_region_masks(landMask, volMask, regions):
    """Benchmark region mask computation methods."""
    from analyser_processor import precompute_region_masks, LazyRegionMaskCache

    print("\n" + "="*60)
    print("BENCHMARK: Region Mask Computation")
    print("="*60)

    # Benchmark precompute (original)
    start = time.perf_counter()
    tracemalloc.start()
    cache_precompute = precompute_region_masks(landMask, volMask, regions)
    current, peak_precompute = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_precompute = time.perf_counter() - start
    print(f"\nPrecompute all masks:")
    print(f"  Time: {time_precompute:.3f}s")
    print(f"  Peak memory: {peak_precompute / 1024 / 1024:.2f} MB")
    print(f"  Masks created: {len(cache_precompute)}")

    # Benchmark lazy (new)
    start = time.perf_counter()
    tracemalloc.start()
    cache_lazy = LazyRegionMaskCache(landMask, volMask, regions)
    # Access only a few masks (typical usage)
    test_regions = [-1, 0, 14, 37]  # global, arctic, atlantic_0, ATL
    for reg in test_regions:
        _ = cache_lazy[f'land_{reg}']
        _ = cache_lazy[f'vol_{reg}']
    current, peak_lazy = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_lazy = time.perf_counter() - start
    stats = cache_lazy.get_stats()
    print(f"\nLazy cache (accessing {len(test_regions)} regions):")
    print(f"  Time: {time_lazy:.3f}s")
    print(f"  Peak memory: {peak_lazy / 1024 / 1024:.2f} MB")
    print(f"  Masks computed: {stats['cached_masks']}")
    print(f"  Masks possible: {stats['possible_masks']}")

    print(f"\nSpeedup: {time_precompute / time_lazy:.1f}x")
    print(f"Memory reduction: {(1 - peak_lazy / peak_precompute) * 100:.1f}%")


def benchmark_subDomainORCA():
    """Benchmark subDomainORCA function."""
    from analyser_functions import subDomainORCA

    print("\n" + "="*60)
    print("BENCHMARK: subDomainORCA Function")
    print("="*60)

    # Create test data (typical ORCA2 size: 12 months, 31 depths, 149x182)
    np.random.seed(42)
    data_3d = np.random.rand(12, 149, 182) * 100
    data_4d = np.random.rand(12, 31, 149, 182) * 100
    var_lons = np.linspace(-180, 180, 182).reshape(1, 182).repeat(149, axis=0)
    var_lats = np.linspace(-90, 90, 149).reshape(149, 1).repeat(182, axis=1)
    landMask = np.ones((149, 182))
    landMask[0:10, :] = np.nan  # Simulate land
    volMask = np.ones((31, 149, 182))
    volMask[:, 0:10, :] = np.nan
    missingVal = 1e20
    lonLim = [-60, 60]
    latLim = [-30, 30]

    # Benchmark 3D data
    n_runs = 10
    data_copy = data_3d.copy()
    start = time.perf_counter()
    for _ in range(n_runs):
        data_copy = data_3d.copy()
        _ = subDomainORCA(lonLim, latLim, var_lons, var_lats, data_copy, landMask, volMask, missingVal)
    time_3d = (time.perf_counter() - start) / n_runs
    print(f"\n3D data (12, 149, 182) - {n_runs} runs avg:")
    print(f"  Time per call: {time_3d * 1000:.2f} ms")

    # Benchmark 4D data
    data_copy = data_4d.copy()
    start = time.perf_counter()
    for _ in range(n_runs):
        data_copy = data_4d.copy()
        _ = subDomainORCA(lonLim, latLim, var_lons, var_lats, data_copy, landMask, volMask, missingVal)
    time_4d = (time.perf_counter() - start) / n_runs
    print(f"\n4D data (12, 31, 149, 182) - {n_runs} runs avg:")
    print(f"  Time per call: {time_4d * 1000:.2f} ms")


def benchmark_data_functions():
    """Benchmark core data processing functions."""
    from analyser_functions import surfaceData, volumeData, levelData, integrateData

    print("\n" + "="*60)
    print("BENCHMARK: Data Processing Functions")
    print("="*60)

    # Create test data
    np.random.seed(42)
    data_3d = np.random.rand(12, 149, 182) * 100
    data_4d = np.random.rand(12, 31, 149, 182) * 100
    var_lons = np.linspace(-180, 180, 182).reshape(1, 182).repeat(149, axis=0)
    var_lats = np.linspace(-90, 90, 149).reshape(149, 1).repeat(182, axis=1)
    area = np.ones((149, 182)) * 1e10
    vol = np.ones((31, 149, 182)) * 1e12
    landMask = np.ones((149, 182))
    volMask = np.ones((31, 149, 182))
    missingVal = 1e20
    lonLim = [-180, 180]
    latLim = [-90, 90]
    units = 1e-15

    n_runs = 5

    # surfaceData
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = surfaceData(data_3d.copy(), var_lons, var_lats, units, area, landMask, volMask, missingVal, lonLim, latLim)
    time_surface = (time.perf_counter() - start) / n_runs
    print(f"\nsurfaceData (3D): {time_surface * 1000:.2f} ms")

    # volumeData
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = volumeData(data_4d.copy(), var_lons, var_lats, units, vol, landMask, volMask, missingVal, lonLim, latLim)
    time_volume = (time.perf_counter() - start) / n_runs
    print(f"volumeData (4D): {time_volume * 1000:.2f} ms")

    # levelData
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = levelData(data_4d.copy(), var_lons, var_lats, units, area, landMask, volMask, missingVal, lonLim, latLim, 0)
    time_level = (time.perf_counter() - start) / n_runs
    print(f"levelData (4D, level=0): {time_level * 1000:.2f} ms")

    # integrateData
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = integrateData(data_4d.copy(), var_lons, var_lats, 0, 10, units, vol, landMask, volMask, missingVal, lonLim, latLim)
    time_integrate = (time.perf_counter() - start) / n_runs
    print(f"integrateData (4D, 0-10): {time_integrate * 1000:.2f} ms")


def benchmark_file_loading(year_from, year_to):
    """Benchmark file loading methods."""
    from analyser_io import load_netcdf_files, HAS_XARRAY

    print("\n" + "="*60)
    print("BENCHMARK: File Loading")
    print("="*60)

    # netCDF4 loading
    start = time.perf_counter()
    tracemalloc.start()
    nc_files, nc_names, years, failed = load_netcdf_files(year_from, year_to)
    current, peak_nc4 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_nc4 = time.perf_counter() - start

    if len(nc_files) > 0:
        print(f"\nnetCDF4 loading ({len(years)} years):")
        print(f"  Time: {time_nc4:.3f}s")
        print(f"  Peak memory: {peak_nc4 / 1024 / 1024:.2f} MB")
        print(f"  Files loaded: {sum(len(f) for f in nc_files)}")

    if HAS_XARRAY:
        from analyser_io import load_netcdf_files_xarray

        start = time.perf_counter()
        tracemalloc.start()
        xr_files, xr_names, years, failed = load_netcdf_files_xarray(year_from, year_to)
        current, peak_xr = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        time_xr = time.perf_counter() - start

        if len(xr_files) > 0:
            print(f"\nxarray loading with Dask ({len(years)} years):")
            print(f"  Time: {time_xr:.3f}s")
            print(f"  Peak memory: {peak_xr / 1024 / 1024:.2f} MB")
            print(f"  Files loaded: {sum(len(f) for f in xr_files)}")

            if len(nc_files) > 0:
                print(f"\nxarray speedup: {time_nc4 / time_xr:.1f}x")
                print(f"Memory change: {(peak_xr / peak_nc4 - 1) * 100:+.1f}%")
    else:
        print("\nxarray not available - skipping xarray benchmark")


def main():
    parser = argparse.ArgumentParser(description='Benchmark analyser performance')
    parser.add_argument('--year-from', type=int, default=2000,
                        help='Starting year for file loading test')
    parser.add_argument('--year-to', type=int, default=2000,
                        help='Ending year for file loading test')
    parser.add_argument('--skip-files', action='store_true',
                        help='Skip file loading benchmarks')
    args = parser.parse_args()

    print("="*60)
    print("ANALYSER PERFORMANCE BENCHMARK")
    print("="*60)

    # Run synthetic benchmarks
    benchmark_subDomainORCA()
    benchmark_data_functions()

    # Create synthetic mask data for region mask benchmark
    np.random.seed(42)
    landMask = np.ones((149, 182))
    landMask[0:10, :] = np.nan
    volMask = np.ones((31, 149, 182))
    volMask[:, 0:10, :] = np.nan
    regions = [np.ones((149, 182)) for _ in range(55)]
    for i, r in enumerate(regions):
        r[i*2:(i+1)*2, :] = np.nan  # Simulate different regions

    benchmark_region_masks(landMask, volMask, regions)

    # File loading benchmark (requires actual files)
    if not args.skip_files:
        benchmark_file_loading(args.year_from, args.year_to)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
