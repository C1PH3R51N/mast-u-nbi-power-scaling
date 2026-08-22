from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

BASE_FOLDER = Path(__file__).parent

OUTPUT_FOLDER = BASE_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

QA_FOLDER = OUTPUT_FOLDER / "segmentation_qa"
QA_FOLDER.mkdir(exist_ok=True)


# Initial segmentation thresholds
PLASMA_CURRENT_THRESHOLD_KA = 100.0
NBI_POWER_THRESHOLD_MW = 0.10

# Beam-specific threshold
BEAM_POWER_THRESHOLD_MW = 0.10


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
        "ANU_NEUTRONS"
    ]
}


# =========================================================
# HELPERS
# =========================================================

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


def get_dimension_coordinate(hdf, dataset):

    dimension_list = dataset.attrs.get(
        "DIMENSION_LIST"
    )

    if dimension_list is None:
        return None

    try:

        for dimension in dimension_list:

            references = np.atleast_1d(
                dimension
            )

            for reference in references:

                coordinate = hdf[reference]

                if (
                    isinstance(
                        coordinate,
                        h5py.Dataset
                    )
                    and coordinate.shape[0]
                    == dataset.shape[0]
                ):

                    return np.asarray(
                        coordinate[()],
                        dtype=float
                    ).ravel()

    except Exception:
        pass

    return None


def extract_signal(file_path, target_signal):

    result = None

    with h5py.File(
        file_path,
        "r"
    ) as hdf:

        def visitor(name, obj):

            nonlocal result

            if result is not None:
                return

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
                key: clean_attribute(value)
                for key, value
                in obj.attrs.items()
            }

            signal_name = attrs.get(
                "original_signal_name",
                ""
            )

            if signal_name != target_signal:
                return

            values = np.asarray(
                obj[()],
                dtype=float
            ).ravel()

            time = get_dimension_coordinate(
                hdf,
                obj
            )

            result = {
                "time": time,
                "values": values,
                "units": attrs.get(
                    "units",
                    ""
                ),
                "label": attrs.get(
                    "label",
                    ""
                )
            }

        hdf.visititems(visitor)

    return result


# =========================================================
# EVENT DETECTION
# =========================================================

def detect_active_window(
    time,
    values,
    threshold
):
    """
    Return first and last times where values exceed threshold.
    """

    mask = (
        np.isfinite(values)
        & (values > threshold)
    )

    if not np.any(mask):
        return None, None

    active_times = time[mask]

    return (
        float(active_times[0]),
        float(active_times[-1])
    )


def calculate_window_metrics(
    time,
    values,
    start,
    end
):
    """
    Calculate metrics inside a defined time window.
    """

    if start is None or end is None:

        return {
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "std_dev": np.nan,
            "samples": 0
        }

    mask = (
        (time >= start)
        & (time <= end)
        & np.isfinite(values)
    )

    selected = values[mask]

    if len(selected) == 0:

        return {
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "std_dev": np.nan,
            "samples": 0
        }

    return {
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
        "std_dev": float(
            np.std(
                selected,
                ddof=1
            )
        ) if len(selected) > 1 else 0.0,
        "samples": len(selected)
    }


def integrate_signal(
    time,
    values,
    start,
    end
):
    """
    Numerically integrate a signal over the selected window.
    """

    if start is None or end is None:
        return np.nan

    mask = (
        (time >= start)
        & (time <= end)
        & np.isfinite(values)
    )

    selected_time = time[mask]
    selected_values = values[mask]

    if len(selected_values) < 2:
        return np.nan

    return float(
        np.trapezoid(
            selected_values,
            selected_time
        )
    )


def classify_beam_mode(
    ss_peak,
    sw_peak
):

    ss_active = (
        np.isfinite(ss_peak)
        and ss_peak > BEAM_POWER_THRESHOLD_MW
    )

    sw_active = (
        np.isfinite(sw_peak)
        and sw_peak > BEAM_POWER_THRESHOLD_MW
    )

    if ss_active and sw_active:
        return "SS + SW"

    if ss_active:
        return "SS only"

    if sw_active:
        return "SW only"

    return "No NBI"


# =========================================================
# DISCOVER FILES
# =========================================================

nc_files = sorted(
    BASE_FOLDER.rglob("*.nc")
)

shot_files = {}


for file_path in nc_files:

    diagnostic, shot = normalise_filename(
        file_path.name
    )

    if diagnostic not in TARGET_SIGNALS:
        continue

    shot_files.setdefault(
        shot,
        {}
    )

    shot_files[shot][
        diagnostic
    ] = file_path


print()
print(
    f"Shots discovered: "
    f"{len(shot_files)}"
)
print()


# =========================================================
# PROCESS SHOTS
# =========================================================

results = []


for shot in sorted(
    shot_files.keys()
):

    print(
        f"Processing shot {shot}..."
    )

    files = shot_files[shot]

    if not all(
        diagnostic in files
        for diagnostic
        in ["AMC", "ANB", "ANU"]
    ):

        print(
            "  Missing core diagnostic - skipped"
        )

        continue


    # -----------------------------------------------------
    # LOAD SIGNALS
    # -----------------------------------------------------

    plasma = extract_signal(
        files["AMC"],
        "AMC_PLASMA CURRENT"
    )

    total_nbi = extract_signal(
        files["ANB"],
        "ANB_TOT_SUM_POWER"
    )

    ss_nbi = extract_signal(
        files["ANB"],
        "ANB_SS_SUM_POWER"
    )

    sw_nbi = extract_signal(
        files["ANB"],
        "ANB_SW_SUM_POWER"
    )

    neutrons = extract_signal(
        files["ANU"],
        "ANU_NEUTRONS"
    )


    if any(
        item is None
        for item in [
            plasma,
            total_nbi,
            ss_nbi,
            sw_nbi,
            neutrons
        ]
    ):

        print(
            "  Missing required signal - skipped"
        )

        continue


    # -----------------------------------------------------
    # DETECT PLASMA WINDOW
    # -----------------------------------------------------

    plasma_start, plasma_end = (
        detect_active_window(
            plasma["time"],
            plasma["values"],
            PLASMA_CURRENT_THRESHOLD_KA
        )
    )


    # -----------------------------------------------------
    # DETECT NBI WINDOW
    # -----------------------------------------------------

    nbi_start, nbi_end = (
        detect_active_window(
            total_nbi["time"],
            total_nbi["values"],
            NBI_POWER_THRESHOLD_MW
        )
    )


    # -----------------------------------------------------
    # PLASMA METRICS
    # -----------------------------------------------------

    plasma_metrics = (
        calculate_window_metrics(
            plasma["time"],
            plasma["values"],
            plasma_start,
            plasma_end
        )
    )


    # -----------------------------------------------------
    # NBI METRICS
    # -----------------------------------------------------

    nbi_metrics = (
        calculate_window_metrics(
            total_nbi["time"],
            total_nbi["values"],
            nbi_start,
            nbi_end
        )
    )


    ss_metrics = (
        calculate_window_metrics(
            ss_nbi["time"],
            ss_nbi["values"],
            nbi_start,
            nbi_end
        )
    )


    sw_metrics = (
        calculate_window_metrics(
            sw_nbi["time"],
            sw_nbi["values"],
            nbi_start,
            nbi_end
        )
    )


    # -----------------------------------------------------
    # NEUTRON METRICS
    # -----------------------------------------------------

    neutron_metrics = (
        calculate_window_metrics(
            neutrons["time"],
            neutrons["values"],
            plasma_start,
            plasma_end
        )
    )


    neutron_integral = (
        integrate_signal(
            neutrons["time"],
            neutrons["values"],
            plasma_start,
            plasma_end
        )
    )


    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    beam_mode = classify_beam_mode(
        ss_metrics["maximum"],
        sw_metrics["maximum"]
    )


    if beam_mode == "No NBI":

        operating_class = (
            "Reference / No NBI"
        )

    elif beam_mode == "SS + SW":

        operating_class = (
            "High Power / Dual Beam"
        )

    else:

        operating_class = (
            "Single Beam NBI"
        )


    # -----------------------------------------------------
    # RECORD RESULT
    # -----------------------------------------------------

    results.append({

        "shot":
            shot,

        "operating_class":
            operating_class,

        "beam_mode":
            beam_mode,

        "plasma_start_s":
            plasma_start,

        "plasma_end_s":
            plasma_end,

        "plasma_duration_s":
            (
                plasma_end
                - plasma_start
                if plasma_start is not None
                and plasma_end is not None
                else np.nan
            ),

        "peak_plasma_current_kA":
            plasma_metrics["maximum"],

        "mean_plasma_current_kA":
            plasma_metrics["mean"],

        "nbi_start_s":
            nbi_start,

        "nbi_end_s":
            nbi_end,

        "nbi_duration_s":
            (
                nbi_end
                - nbi_start
                if nbi_start is not None
                and nbi_end is not None
                else 0
            ),

        "peak_total_nbi_MW":
            nbi_metrics["maximum"],

        "mean_total_nbi_MW":
            nbi_metrics["mean"],

        "peak_ss_nbi_MW":
            ss_metrics["maximum"],

        "peak_sw_nbi_MW":
            sw_metrics["maximum"],

        "peak_neutron_rate_n_s":
            neutron_metrics["maximum"],

        "mean_neutron_rate_n_s":
            neutron_metrics["mean"],

        "integrated_neutron_output":
            neutron_integral
    })


    # =====================================================
    # QA PLOT
    # =====================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )


    # Plasma
    axes[0].plot(
        plasma["time"],
        plasma["values"]
    )

    axes[0].axhline(
        PLASMA_CURRENT_THRESHOLD_KA,
        linestyle="--"
    )

    if plasma_start is not None:

        axes[0].axvline(
            plasma_start,
            linestyle="--"
        )

    if plasma_end is not None:

        axes[0].axvline(
            plasma_end,
            linestyle="--"
        )

    axes[0].set_ylabel(
        "Plasma Current\n(kA)"
    )


    # NBI
    axes[1].plot(
        total_nbi["time"],
        total_nbi["values"]
    )

    axes[1].axhline(
        NBI_POWER_THRESHOLD_MW,
        linestyle="--"
    )

    if nbi_start is not None:

        axes[1].axvline(
            nbi_start,
            linestyle="--"
        )

    if nbi_end is not None:

        axes[1].axvline(
            nbi_end,
            linestyle="--"
        )

    axes[1].set_ylabel(
        "Total NBI\n(MW)"
    )


    # Neutrons
    axes[2].plot(
        neutrons["time"],
        neutrons["values"]
    )

    if plasma_start is not None:

        axes[2].axvline(
            plasma_start,
            linestyle="--"
        )

    if plasma_end is not None:

        axes[2].axvline(
            plasma_end,
            linestyle="--"
        )

    axes[2].set_ylabel(
        "Neutron Rate\n(n/s)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )


    fig.suptitle(
        f"Shot {shot} — "
        f"{operating_class} | "
        f"{beam_mode}",
        fontsize=14,
        fontweight="bold"
    )


    for axis in axes:

        axis.grid(
            alpha=0.25
        )

        axis.set_xlim(
            -0.1,
            0.7
        )


    fig.tight_layout()


    fig.savefig(
        QA_FOLDER
        / f"shot_{shot}_segmentation_qa.png",
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# =========================================================
# SAVE SHOT METRICS
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "shot"
)

results_df.to_csv(
    OUTPUT_FOLDER
    / "shot_metrics_v1.csv",
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print(
    "V3C SHOT SEGMENTATION COMPLETE"
)
print("=" * 60)

print(
    f"Shots processed: "
    f"{len(results_df)}"
)

print()
print("Operating classes:")
print()

print(
    results_df[
        "operating_class"
    ]
    .value_counts()
    .to_string()
)

print()
print("Beam modes:")
print()

print(
    results_df[
        "beam_mode"
    ]
    .value_counts()
    .to_string()
)

print()
print("Outputs:")
print(
    OUTPUT_FOLDER
    / "shot_metrics_v1.csv"
)

print(
    QA_FOLDER
)