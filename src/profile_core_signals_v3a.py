from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_FOLDER = Path(__file__).parent
OUTPUT_FOLDER = BASE_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)


TARGET_SIGNALS = {

    "AMC": [
        "AMC_PLASMA CURRENT"
    ],

    "ANB": [
        "ANB_TOT_SUM_POWER",
        "ANB_SS_SUM_POWER",
        "ANB_SW_SUM_POWER"
    ],

    "ANU": [
        "ANU_NEUTRONS",
        "ANU_NEUTRONS_DC",
        "ANU_NEUTRONS_CB",
        "ANU_ERRORS"
    ]
}


# =========================================================
# HELPERS
# =========================================================

def clean_attribute(value):

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace"
        )

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list) and len(value) == 1:
        return value[0]

    return value


def normalise_filename(filename):

    match = re.search(
        r"([a-z]{3})(\d{5})",
        filename.lower()
    )

    if not match:
        return None, None

    return (
        match.group(1).upper(),
        int(match.group(2))
    )


def safe_numeric_array(dataset):

    try:

        values = np.asarray(
            dataset[()]
        )

        if not np.issubdtype(
            values.dtype,
            np.number
        ):
            return None

        return values.astype(
            float
        ).ravel()

    except Exception:
        return None


# =========================================================
# TIME AXIS DISCOVERY
# =========================================================

def find_time_dataset(
    hdf,
    data_path,
    data_dataset
):
    """
    Locate the time coordinate belonging to a signal.

    Search order:

    1. HDF5 DIMENSION_LIST reference
    2. Time dataset inside the signal's parent group
    3. Root-level /time dataset with matching length

    Returns:

        time_path
        time_dataset
        discovery_method
    """

    # -----------------------------------------------------
    # METHOD 1
    # HDF5 dimension reference
    # -----------------------------------------------------

    try:

        dimension_list = (
            data_dataset.attrs.get(
                "DIMENSION_LIST"
            )
        )

        if dimension_list is not None:

            for dimension in dimension_list:

                references = np.atleast_1d(
                    dimension
                )

                for reference in references:

                    try:

                        coordinate = hdf[
                            reference
                        ]

                        if (
                            isinstance(
                                coordinate,
                                h5py.Dataset
                            )
                            and coordinate.shape[0]
                            == data_dataset.shape[0]
                        ):

                            return (
                                coordinate.name,
                                coordinate,
                                "HDF5 dimension reference"
                            )

                    except Exception:
                        pass

    except Exception:
        pass


    # -----------------------------------------------------
    # METHOD 2
    # Local sibling time dataset
    # -----------------------------------------------------

    parent_path = (
        data_path.rsplit("/", 1)[0]
    )

    local_candidates = [

        f"{parent_path}/time",
        f"{parent_path}/times",
        f"{parent_path}/t"

    ]

    for candidate in local_candidates:

        if candidate in hdf:

            coordinate = hdf[
                candidate
            ]

            if (
                isinstance(
                    coordinate,
                    h5py.Dataset
                )
                and coordinate.shape[0]
                == data_dataset.shape[0]
            ):

                return (
                    coordinate.name,
                    coordinate,
                    "Local time dataset"
                )


    # -----------------------------------------------------
    # METHOD 3
    # Root-level time dataset
    # -----------------------------------------------------

    if "time" in hdf:

        coordinate = hdf["time"]

        if (
            isinstance(
                coordinate,
                h5py.Dataset
            )
            and coordinate.shape[0]
            == data_dataset.shape[0]
        ):

            return (
                coordinate.name,
                coordinate,
                "Root time dataset"
            )


    # -----------------------------------------------------
    # NOTHING FOUND
    # -----------------------------------------------------

    return (
        None,
        None,
        "Not found"
    )


# =========================================================
# NUMERICAL PROFILING
# =========================================================

def calculate_numeric_profile(values):

    if values is None or len(values) == 0:

        return {
            "observations": 0,
            "finite_observations": 0,
            "nan_count": 0,
            "inf_count": 0,
            "zero_count": 0,
            "negative_count": 0,
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std_dev": np.nan
        }


    nan_count = int(
        np.isnan(values).sum()
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    finite_values = values[
        np.isfinite(values)
    ]


    if len(finite_values) == 0:

        return {
            "observations": len(values),
            "finite_observations": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_count": 0,
            "negative_count": 0,
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std_dev": np.nan
        }


    return {

        "observations":
            len(values),

        "finite_observations":
            len(finite_values),

        "nan_count":
            nan_count,

        "inf_count":
            inf_count,

        "zero_count":
            int(
                (finite_values == 0).sum()
            ),

        "negative_count":
            int(
                (finite_values < 0).sum()
            ),

        "minimum":
            float(
                np.min(finite_values)
            ),

        "maximum":
            float(
                np.max(finite_values)
            ),

        "mean":
            float(
                np.mean(finite_values)
            ),

        "median":
            float(
                np.median(finite_values)
            ),

        "std_dev":
            float(
                np.std(
                    finite_values,
                    ddof=1
                )
            )
            if len(finite_values) > 1
            else 0.0
    }


# =========================================================
# TIME PROFILING
# =========================================================

def calculate_time_profile(
    time_values
):

    if (
        time_values is None
        or len(time_values) == 0
    ):

        return {
            "time_found": False,
            "time_observations": 0,
            "time_start": np.nan,
            "time_end": np.nan,
            "duration_s": np.nan,
            "median_timestep_s": np.nan,
            "min_timestep_s": np.nan,
            "max_timestep_s": np.nan,
            "duplicate_time_count": np.nan
        }


    finite_time = time_values[
        np.isfinite(time_values)
    ]


    if len(finite_time) == 0:

        return {
            "time_found": True,
            "time_observations":
                len(time_values),
            "time_start": np.nan,
            "time_end": np.nan,
            "duration_s": np.nan,
            "median_timestep_s": np.nan,
            "min_timestep_s": np.nan,
            "max_timestep_s": np.nan,
            "duplicate_time_count": np.nan
        }


    sorted_time = np.sort(
        finite_time
    )


    if len(sorted_time) > 1:

        differences = np.diff(
            sorted_time
        )

        median_timestep = float(
            np.median(differences)
        )

        min_timestep = float(
            np.min(differences)
        )

        max_timestep = float(
            np.max(differences)
        )

        duplicate_count = int(
            (differences == 0).sum()
        )

    else:

        median_timestep = np.nan
        min_timestep = np.nan
        max_timestep = np.nan
        duplicate_count = 0


    return {

        "time_found":
            True,

        "time_observations":
            len(time_values),

        "time_start":
            float(
                np.min(finite_time)
            ),

        "time_end":
            float(
                np.max(finite_time)
            ),

        "duration_s":
            float(
                np.max(finite_time)
                - np.min(finite_time)
            ),

        "median_timestep_s":
            median_timestep,

        "min_timestep_s":
            min_timestep,

        "max_timestep_s":
            max_timestep,

        "duplicate_time_count":
            duplicate_count
    }


# =========================================================
# DISCOVER FILES
# =========================================================

nc_files = sorted(
    BASE_FOLDER.rglob("*.nc")
)

print()
print(
    f"Found {len(nc_files)} "
    f"NetCDF/HDF5 files."
)

if len(nc_files) == 0:

    print(
        "ERROR: No .nc files found."
    )

    raise SystemExit


# =========================================================
# PROFILE CORE SIGNALS
# =========================================================

profiles = []

files_checked = 0


for file_path in nc_files:

    diagnostic, shot = (
        normalise_filename(
            file_path.name
        )
    )


    if diagnostic not in TARGET_SIGNALS:
        continue


    files_checked += 1

    print(
        f"Shot {shot} | "
        f"{diagnostic} | "
        f"{file_path.name}"
    )


    try:

        with h5py.File(
            file_path,
            "r"
        ) as hdf:


            def visitor(name, obj):

                if not isinstance(
                    obj,
                    h5py.Dataset
                ):
                    return


                if not name.endswith(
                    "/data"
                ):
                    return


                attrs = {

                    key:
                    clean_attribute(
                        value
                    )

                    for key, value
                    in obj.attrs.items()

                }


                signal_name = attrs.get(
                    "original_signal_name",
                    name
                )


                if signal_name not in (
                    TARGET_SIGNALS[
                        diagnostic
                    ]
                ):
                    return


                # -----------------------------------------
                # SIGNAL METADATA
                # -----------------------------------------

                label = attrs.get(
                    "label",
                    ""
                )

                units = attrs.get(
                    "units",
                    ""
                )


                # -----------------------------------------
                # DATA
                # -----------------------------------------

                values = (
                    safe_numeric_array(
                        obj
                    )
                )

                numeric_profile = (
                    calculate_numeric_profile(
                        values
                    )
                )


                # -----------------------------------------
                # TIME
                # -----------------------------------------

                (
                    time_path,
                    time_dataset,
                    time_method

                ) = find_time_dataset(

                    hdf,
                    name,
                    obj

                )


                if time_dataset is not None:

                    time_values = (
                        safe_numeric_array(
                            time_dataset
                        )
                    )

                else:

                    time_values = None


                time_profile = (
                    calculate_time_profile(
                        time_values
                    )
                )


                # -----------------------------------------
                # LENGTH CHECK
                # -----------------------------------------

                if (
                    values is not None
                    and time_values is not None
                ):

                    length_match = (
                        len(values)
                        == len(time_values)
                    )

                else:

                    length_match = False


                # -----------------------------------------
                # RECORD
                # -----------------------------------------

                row = {

                    "shot":
                        shot,

                    "diagnostic":
                        diagnostic,

                    "signal_name":
                        signal_name,

                    "label":
                        label,

                    "units":
                        units,

                    "source_file":
                        file_path.name,

                    "dataset_path":
                        name,

                    "time_dataset_path":
                        time_path,

                    "time_discovery_method":
                        time_method,

                    "data_time_length_match":
                        length_match
                }


                row.update(
                    numeric_profile
                )

                row.update(
                    time_profile
                )


                profiles.append(
                    row
                )


            hdf.visititems(
                visitor
            )


    except Exception as error:

        print(
            f"ERROR: {error}"
        )


# =========================================================
# DATAFRAME
# =========================================================

profile_df = pd.DataFrame(
    profiles
)


if profile_df.empty:

    print(
        "No target signals found."
    )

    raise SystemExit


# =========================================================
# QA FIELDS
# =========================================================

profile_df[
    "missing_pct"
] = (

    profile_df["nan_count"]
    / profile_df["observations"]
    * 100

).round(4)


profile_df[
    "finite_pct"
] = (

    profile_df[
        "finite_observations"
    ]
    / profile_df["observations"]
    * 100

).round(4)


def time_quality(row):

    if not row["time_found"]:

        return "No time axis"


    if not row[
        "data_time_length_match"
    ]:

        return "Length mismatch"


    if row[
        "duplicate_time_count"
    ] > 0:

        return "Duplicate timestamps"


    return "OK"


profile_df[
    "time_quality"
] = profile_df.apply(
    time_quality,
    axis=1
)


# =========================================================
# SHOT SUMMARY
# =========================================================

shot_summary = (

    profile_df
    .groupby("shot")
    .agg(

        signals_profiled=(
            "signal_name",
            "nunique"
        ),

        diagnostics_profiled=(
            "diagnostic",
            "nunique"
        ),

        signals_with_time=(
            "time_found",
            "sum"
        ),

        signals_time_aligned=(
            "data_time_length_match",
            "sum"
        ),

        total_nan=(
            "nan_count",
            "sum"
        ),

        total_inf=(
            "inf_count",
            "sum"
        )
    )

    .reset_index()
)


# =========================================================
# SIGNAL SUMMARY
# =========================================================

signal_summary = (

    profile_df

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

        shots_available=(
            "shot",
            "nunique"
        ),

        mean_observations=(
            "observations",
            "mean"
        ),

        mean_missing_pct=(
            "missing_pct",
            "mean"
        ),

        minimum_value=(
            "minimum",
            "min"
        ),

        maximum_value=(
            "maximum",
            "max"
        ),

        mean_value_across_shots=(
            "mean",
            "mean"
        ),

        median_timestep_s=(
            "median_timestep_s",
            "median"
        ),

        earliest_time=(
            "time_start",
            "min"
        ),

        latest_time=(
            "time_end",
            "max"
        )
    )

    .reset_index()
)


# =========================================================
# SORT
# =========================================================

profile_df = (
    profile_df.sort_values(
        [
            "shot",
            "diagnostic",
            "signal_name"
        ]
    )
)

shot_summary = (
    shot_summary.sort_values(
        "shot"
    )
)

signal_summary = (
    signal_summary.sort_values(
        [
            "diagnostic",
            "signal_name"
        ]
    )
)


# =========================================================
# SAVE OUTPUTS
# =========================================================

profile_df.to_csv(

    OUTPUT_FOLDER
    / "core_signal_numeric_profile.csv",

    index=False
)


shot_summary.to_csv(

    OUTPUT_FOLDER
    / "core_signal_shot_summary.csv",

    index=False
)


signal_summary.to_csv(

    OUTPUT_FOLDER
    / "core_signal_summary.csv",

    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("=" * 60)
print(
    "V3A CORE SIGNAL PROFILING COMPLETE"
)
print("=" * 60)

print(
    f"Files checked:          "
    f"{files_checked}"
)

print(
    f"Signal records:         "
    f"{len(profile_df)}"
)

print(
    f"Shots:                  "
    f"{profile_df['shot'].nunique()}"
)

print(
    f"Diagnostics:            "
    f"{profile_df['diagnostic'].nunique()}"
)


print()
print("TIME DISCOVERY METHODS")
print("-" * 60)

print(
    profile_df[
        "time_discovery_method"
    ]
    .value_counts()
    .to_string()
)


print()
print("TIME QUALITY")
print("-" * 60)

print(
    profile_df[
        "time_quality"
    ]
    .value_counts()
    .to_string()
)


print()
print(
    "Outputs saved to:"
)

print(
    OUTPUT_FOLDER.resolve()
)
