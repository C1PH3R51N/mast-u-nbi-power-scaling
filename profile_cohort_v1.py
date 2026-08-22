from pathlib import Path
import re

import h5py
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Use the folder that this Python script is stored in.
# The script will search all subfolders recursively for .nc files.
COHORT_FOLDER = Path(__file__).parent

# Create an outputs folder beside the script.
OUTPUT_FOLDER = COHORT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_attribute(value):
    """
    Convert HDF5 attributes into normal Python-readable values.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list) and len(value) == 1:
        return value[0]

    return value


def normalise_filename(filename):
    """
    Recognises filenames such as:

        anu27929.nc
        anu27930 (1).nc
        amc27930 (2).nc

    Returns:
        diagnostic code
        shot number
    """

    match = re.search(r"([a-z]{3})(\d{5})", filename.lower())

    if not match:
        return None, None

    diagnostic = match.group(1).upper()
    shot = int(match.group(2))

    return diagnostic, shot


def get_root_metadata(hdf_file):
    """
    Extract metadata stored at the root of the HDF5 file.
    """

    metadata = {}

    for key, value in hdf_file.attrs.items():
        metadata[key] = clean_attribute(value)

    return metadata


# ---------------------------------------------------------
# SIGNAL DISCOVERY
# ---------------------------------------------------------

def extract_signals(file_path, shot, diagnostic):
    """
    Search a MAST NetCDF/HDF5 file for measurement datasets.

    Datasets ending with '/data' are treated as signal arrays.
    """

    signal_rows = []

    with h5py.File(file_path, "r") as hdf:

        def visitor(name, obj):

            if not isinstance(obj, h5py.Dataset):
                return

            if not name.endswith("/data"):
                return

            attrs = {
                key: clean_attribute(value)
                for key, value in obj.attrs.items()
            }

            original_signal = attrs.get(
                "original_signal_name",
                name
            )

            label = attrs.get("label", "")
            units = attrs.get("units", "")

            signal_rows.append({
                "shot": shot,
                "diagnostic": diagnostic,
                "signal_name": original_signal,
                "dataset_path": name,
                "label": label,
                "units": units,
                "shape": str(obj.shape),
                "dimensions": len(obj.shape),
                "observations": obj.size,
                "dtype": str(obj.dtype),
                "source_file": file_path.name
            })

        hdf.visititems(visitor)

    return signal_rows


# ---------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------

file_inventory = []
signal_inventory = []

nc_files = sorted(COHORT_FOLDER.rglob("*.nc"))

print(f"Searching folder:")
print(COHORT_FOLDER)
print()

print(f"Found {len(nc_files)} NetCDF/HDF5 files.")
print()


# Stop cleanly if no files are found.
if len(nc_files) == 0:
    print("ERROR: No .nc files were found.")
    print("Check that the shot folders are stored beneath:")
    print(COHORT_FOLDER)
    raise SystemExit


# ---------------------------------------------------------
# PROFILE EACH FILE
# ---------------------------------------------------------

for file_number, file_path in enumerate(nc_files, start=1):

    diagnostic, filename_shot = normalise_filename(file_path.name)

    print(
        f"[{file_number}/{len(nc_files)}] "
        f"{file_path.name}"
    )

    try:

        with h5py.File(file_path, "r") as hdf:

            metadata = get_root_metadata(hdf)

            metadata_shot = metadata.get("shot")
            title = metadata.get("title")
            comment = metadata.get("comment")
            data_class = metadata.get("class")
            date = metadata.get("date")
            status = metadata.get("status")

        # Prefer the shot number stored inside the file.
        # Fall back to the shot number extracted from the filename.
        try:
            shot = int(metadata_shot)
        except (TypeError, ValueError):
            shot = filename_shot

        file_inventory.append({
            "shot": shot,
            "diagnostic": diagnostic,
            "filename": file_path.name,
            "relative_path": str(
                file_path.relative_to(COHORT_FOLDER)
            ),
            "diagnostic_title": title,
            "diagnostic_description": comment,
            "data_class": data_class,
            "date": date,
            "status": status,
            "readable": True,
            "error": ""
        })

        signals = extract_signals(
            file_path,
            shot,
            diagnostic
        )

        signal_inventory.extend(signals)

    except Exception as error:

        file_inventory.append({
            "shot": filename_shot,
            "diagnostic": diagnostic,
            "filename": file_path.name,
            "relative_path": str(
                file_path.relative_to(COHORT_FOLDER)
            ),
            "diagnostic_title": None,
            "diagnostic_description": None,
            "data_class": None,
            "date": None,
            "status": None,
            "readable": False,
            "error": str(error)
        })


# ---------------------------------------------------------
# CREATE DATAFRAMES
# ---------------------------------------------------------

files_df = pd.DataFrame(file_inventory)
signals_df = pd.DataFrame(signal_inventory)

total_shots = files_df["shot"].dropna().nunique()


# ---------------------------------------------------------
# DIAGNOSTIC COVERAGE
# ---------------------------------------------------------

diagnostic_df = (
    files_df[
        files_df["readable"] == True
    ]
    .groupby("diagnostic")
    .agg(
        diagnostic_description=(
            "diagnostic_description",
            "first"
        ),
        shots_available=(
            "shot",
            "nunique"
        ),
        files_found=(
            "filename",
            "count"
        )
    )
    .reset_index()
)

diagnostic_df["total_cohort_shots"] = total_shots

diagnostic_df["coverage_pct"] = (
    diagnostic_df["shots_available"]
    / total_shots
    * 100
).round(1)


# ---------------------------------------------------------
# SIGNAL COVERAGE
# ---------------------------------------------------------

if not signals_df.empty:

    signal_coverage = (
        signals_df
        .groupby(
            [
                "diagnostic",
                "signal_name",
                "label",
                "units"
            ],
            dropna=False
        )
        .agg(
            shots_available=("shot", "nunique"),
            example_shape=("shape", "first"),
            example_dtype=("dtype", "first")
        )
        .reset_index()
    )

    signal_coverage["total_cohort_shots"] = total_shots

    signal_coverage["coverage_pct"] = (
        signal_coverage["shots_available"]
        / total_shots
        * 100
    ).round(1)

else:

    signal_coverage = pd.DataFrame()


# ---------------------------------------------------------
# SORT RESULTS
# ---------------------------------------------------------

files_df = files_df.sort_values(
    ["shot", "diagnostic", "filename"]
)

diagnostic_df = diagnostic_df.sort_values(
    ["coverage_pct", "diagnostic"],
    ascending=[False, True]
)

if not signals_df.empty:

    signals_df = signals_df.sort_values(
        ["diagnostic", "signal_name", "shot"]
    )

if not signal_coverage.empty:

    signal_coverage = signal_coverage.sort_values(
        ["coverage_pct", "diagnostic", "signal_name"],
        ascending=[False, True, True]
    )


# ---------------------------------------------------------
# SAVE OUTPUT FILES
# ---------------------------------------------------------

files_df.to_csv(
    OUTPUT_FOLDER / "cohort_file_inventory.csv",
    index=False
)

diagnostic_df.to_csv(
    OUTPUT_FOLDER / "diagnostic_catalogue.csv",
    index=False
)

signals_df.to_csv(
    OUTPUT_FOLDER / "signal_inventory.csv",
    index=False
)

signal_coverage.to_csv(
    OUTPUT_FOLDER / "signal_catalogue.csv",
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("----------------------------------------")
print("COHORT PROFILE COMPLETE")
print("----------------------------------------")

print(f"Shots discovered:       {total_shots}")
print(f"Files discovered:       {len(files_df)}")

print(
    f"Readable files:         "
    f"{files_df['readable'].sum()}"
)

print(
    f"Unreadable files:       "
    f"{(~files_df['readable']).sum()}"
)

print(
    f"Diagnostics discovered: "
    f"{diagnostic_df['diagnostic'].nunique()}"
)

if not signals_df.empty:
    print(
        f"Unique signals found:   "
        f"{signals_df['signal_name'].nunique()}"
    )
else:
    print("Unique signals found:   0")

print()
print("Diagnostic coverage:")
print()

print(
    diagnostic_df[
        [
            "diagnostic",
            "shots_available",
            "coverage_pct"
        ]
    ].to_string(index=False)
)

print()
print("Output files saved to:")
print(OUTPUT_FOLDER.resolve())