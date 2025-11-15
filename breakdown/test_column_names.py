#!/usr/bin/env python3
"""Quick test to verify column names are parsed from TOML."""

from breakdown_config import parse_toml_config

config = parse_toml_config('breakdown_config.toml')

print("Surface variables column names:")
for var in config.surface_vars[:5]:
    print(f"  {var.name:15} -> {var.column_name}")

print("\nIntegration variables column names:")
for var in config.integration_vars[:5]:
    print(f"  {var.name:15} -> {var.column_name}")

print("\nLevel variables column names:")
for var in config.level_vars[:5]:
    print(f"  {var.name:15} -> {var.column_name}")
