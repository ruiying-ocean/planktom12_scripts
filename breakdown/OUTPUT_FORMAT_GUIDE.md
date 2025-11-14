# Breakdown Output Format Guide

## New Features

The breakdown system now supports clean CSV output format with configurable options.

## Configuration

Edit these settings in `breakdown.py` (around line 594):

```python
OUTPUT_FORMAT = 'csv'                # 'csv' or 'tsv' (tab-separated)
INCLUDE_UNITS_IN_HEADERS = False     # Include units in column names
INCLUDE_KEYS_IN_HEADERS = False      # Include keys in column names
```

## Output Format Comparison

### Old Format (TSV with 3 header rows)

```
year    BAC     COC     DIA     ...
        Conc->PgCarbon  Conc->PgCarbon  Conc->PgCarbon  ...
        global  global  global  ...
2101    1.1063e-01      2.4383e-02      9.5089e-02      ...
```

**Problems:**
- 3 header rows (hard to parse)
- Units and keys repeat for every column
- Tab-separated (can have alignment issues)
- Duplicate column names when same key used multiple times

### New Format (CSV with single header)

**Option 1: Clean headers (default)**
```csv
year,BAC,COC,DIA,FIX,GEL,CRU,MES,MIX,PHA,PIC,PRO,PTE
2101,1.1063e-01,2.4383e-02,9.5089e-02,9.8074e-02,6.8874e-02,1.4617e-02,2.4173e-01,6.2566e-04,9.9299e-03,3.4574e-01,6.6198e-02,3.1346e-02
```

**Option 2: With units (set `INCLUDE_UNITS_IN_HEADERS = True`)**
```csv
year,BAC (Conc->PgCarbon),COC (Conc->PgCarbon),DIA (Conc->PgCarbon),...
2101,1.1063e-01,2.4383e-02,9.5089e-02,...
```

**Option 3: With keys (set `INCLUDE_KEYS_IN_HEADERS = True`)**
```csv
year,BAC [global],COC [global],DIA [global],...
2101,1.1063e-01,2.4383e-02,9.5089e-02,...
```

**Option 4: With both (set both to `True`)**
```csv
year,BAC (Conc->PgCarbon) [global],COC (Conc->PgCarbon) [global],...
2101,1.1063e-01,2.4383e-02,9.5089e-02,...
```

## Benefits

✅ **Clean, parseable format** - Single header row, CSV standard
✅ **Easy to import** - Opens directly in Excel, pandas, R
✅ **Flexible** - Choose what metadata to include
✅ **Smaller files** - No redundant header rows
✅ **Better for automation** - Standard CSV tools work out of the box

## Output Files

### Currently Generated (CSV mode)

- `breakdown.sur.annual.csv` - Surface variables (annual means)
- `breakdown.vol.annual.csv` - Volume-integrated variables

### Available but Commented Out

Uncomment these lines in `breakdown.py` if you need them:

- `breakdown.lev.annual.csv` - Level-specific variables
- `breakdown.int.annual.csv` - Depth-integrated variables
- `breakdown.ave.annual.csv` - Depth-averaged variables
- `breakdown.obs.annual.dat` - Observation comparisons
- `breakdown.*.spread.dat` - Monthly percentile spreads
- `breakdown.*.monthly.dat` - Monthly time series
- `breakdown.*.*.dat` - Emergent properties (bloom, trophic, etc.)

## Switching Between Formats

### To use CSV format:
```python
OUTPUT_FORMAT = 'csv'
```
Outputs: `*.csv` files with single header row

### To use original TSV format:
```python
OUTPUT_FORMAT = 'tsv'
```
Outputs: `*.dat` files with 3 header rows (backward compatible)

## Reading CSV Files

### Python (pandas)
```python
import pandas as pd
df = pd.read_csv('breakdown.sur.annual.csv')
print(df.head())
```

### R
```r
data <- read.csv('breakdown.sur.annual.csv')
head(data)
```

### Excel
Just double-click the `.csv` file, or:
File → Open → Select CSV file → Import

### Command line
```bash
# View first 10 rows
head breakdown.sur.annual.csv

# Count rows
wc -l breakdown.sur.annual.csv

# Extract specific columns
cut -d, -f1,2,3 breakdown.sur.annual.csv
```

## Migration Notes

If you have existing scripts that read the old `.dat` files:

1. Update file extensions: `.dat` → `.csv`
2. Update parsing: Single header row instead of 3
3. Update separator: Tab → Comma
4. Or set `OUTPUT_FORMAT = 'tsv'` to keep old format

## Customization

To add more outputs, edit `breakdown.py` around line 600:

```python
# Add any output you need:
writer.write_annual_csv("breakdown.int.annual.csv", varInt, year_from, year_to, 0, False, False)
writer.write_annual_csv("breakdown.ave.annual.csv", varTotalAve, year_from, year_to, 0, False, True)  # Include keys
```

To create monthly or spread outputs in CSV format, you can adapt the `write_annual_csv` method or request it as a feature.
