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

QA_FOLDER = OUTPUT_FOLDER / "nbi_state_qa"
QA_FOLDER.mkdir(exist_ok=True)


# Threshold used to decide whether a beam is genuinely on
BEAM_ON_THRESHOLD_MW = 0.10

# Minimum duration for a state to count as a real NBI state
MIN_STATE_DURATION_S = 0.015

# Power-change tolerance used to separate sustained states
POWER_CHANGE_THRESHOLD_MW = 0.40


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

            if not name.endswith("/data"):
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
                "units": attrs.get("units", ""),
                "label": attrs.get("label", "")
            }

        hdf.visititems(visitor)

    return result


# =========================================================
# INTERPOLATION
# =========================================================

def interpolate_to_time(
    source_time,
    source_values,
    target_time
):
    """
    Interpolate one signal onto another signal's timebase.
    """

    finite_mask = (
        np.isfinite(source_time)
        & np.isfinite(source_values)
    )

    source_time = source_time[finite_mask]
    source_values = source_values[finite_mask]

    if len(source_time) < 2:
        return np.full(
            len(target_time),
            np.nan
        )

    return np.interp(
        target_time,
        source_time,
        source_values,
        left=np.nan,
        right=np.nan
    )


# =========================================================
# STATE DETECTION
# =========================================================

def classify_instantaneous_beam_state(
    total_power,
    ss_power,
    sw_power
):

    ss_on = ss_power > BEAM_ON_THRESHOLD_MW
    sw_on = sw_power > BEAM_ON_THRESHOLD_MW

    if ss_on and sw_on:
        return "SS + SW"

    if ss_on:
        return "SS only"

    if sw_on:
        return "SW only"

    return "No NBI"


def build_initial_state_series(
    time,
    total_power,
    ss_power,
    sw_power
):
    """
    Assign an instantaneous state to every NBI sample.
    """

    states = []

    for total, ss, sw in zip(
        total_power,
        ss_power,
        sw_power
    ):

        states.append(
            classify_instantaneous_beam_state(
                total,
                ss,
                sw
            )
        )

    return np.array(
        states,
        dtype=object
    )


def find_contiguous_segments(
    time,
    states
):
    """
    Convert state labels into contiguous time segments.
    """

    segments = []

    if len(states) == 0:
        return segments

    start_index = 0
    current_state = states[0]

    for i in range(1, len(states)):

        if states[i] != current_state:

            segments.append({
                "start_index": start_index,
                "end_index": i - 1,
                "state": current_state
            })

            start_index = i
            current_state = states[i]

    segments.append({
        "start_index": start_index,
        "end_index": len(states) - 1,
        "state": current_state
    })

    return segments


def segment_duration(
    time,
    start_index,
    end_index
):

    return float(
        time[end_index]
        - time[start_index]
    )


def merge_short_segments(
    time,
    segments
):
    """
    Remove very short switching artefacts.

    Short segments are merged into a neighbouring state
    rather than treated as a genuine operating state.
    """

    if len(segments) <= 1:
        return segments

    changed = True

    while changed:

        changed = False
        new_segments = []
        i = 0

        while i < len(segments):

            segment = segments[i]

            duration = segment_duration(
                time,
                segment["start_index"],
                segment["end_index"]
            )

            if (
                duration < MIN_STATE_DURATION_S
                and len(segments) > 1
            ):

                # Prefer merging into previous segment
                if len(new_segments) > 0:

                    new_segments[-1][
                        "end_index"
                    ] = segment["end_index"]

                    changed = True
                    i += 1
                    continue

                # Otherwise merge into next segment
                elif i + 1 < len(segments):

                    segments[i + 1][
                        "start_index"
                    ] = segment["start_index"]

                    changed = True
                    i += 1
                    continue

            new_segments.append(
                segment.copy()
            )

            i += 1

        segments = new_segments

    return segments


def split_on_large_power_change(
    time,
    total_power,
    segments
):
    """
    Split a sustained beam state if total power changes
    by a large amount for a sustained period.

    Useful for cases such as shot 27933:
    ~1.5 MW -> ~3 MW.
    """

    refined = []

    for segment in segments:

        start = segment["start_index"]
        end = segment["end_index"]
        state = segment["state"]

        if state == "No NBI":

            refined.append(
                segment
            )

            continue

        segment_power = total_power[
            start:end + 1
        ]

        segment_time = time[
            start:end + 1
        ]

        if len(segment_power) < 10:

            refined.append(
                segment
            )

            continue

        # Median first and second halves
        midpoint = len(
            segment_power
        ) // 2

        first_median = np.nanmedian(
            segment_power[:midpoint]
        )

        second_median = np.nanmedian(
            segment_power[midpoint:]
        )

        difference = abs(
            second_median
            - first_median
        )

        if difference < POWER_CHANGE_THRESHOLD_MW:

            refined.append(
                segment
            )

            continue

        # Detect largest sustained step
        smoothed = pd.Series(
            segment_power
        ).rolling(
            window=25,
            center=True,
            min_periods=1
        ).median().to_numpy()

        gradient = np.abs(
            np.diff(smoothed)
        )

        split_local = int(
            np.argmax(gradient)
        ) + 1

        split_index = (
            start
            + split_local
        )

        duration_before = (
            time[split_index]
            - time[start]
        )

        duration_after = (
            time[end]
            - time[split_index]
        )

        if (
            duration_before
            >= MIN_STATE_DURATION_S
            and duration_after
            >= MIN_STATE_DURATION_S
        ):

            refined.append({
                "start_index": start,
                "end_index": split_index - 1,
                "state": state
            })

            refined.append({
                "start_index": split_index,
                "end_index": end,
                "state": state
            })

        else:

            refined.append(
                segment
            )

    return refined


# =========================================================
# SEGMENT METRICS
# =========================================================

def calculate_metrics(
    time,
    values,
    start_time,
    end_time
):

    mask = (
        (time >= start_time)
        & (time <= end_time)
        & np.isfinite(values)
    )

    selected_time = time[mask]
    selected_values = values[mask]

    if len(selected_values) == 0:

        return {
            "mean": np.nan,
            "median": np.nan,
            "maximum": np.nan,
            "minimum": np.nan,
            "std_dev": np.nan,
            "integral": np.nan,
            "samples": 0
        }

    integral = np.nan

    if len(selected_values) >= 2:

        integral = float(
            np.trapezoid(
                selected_values,
                selected_time
            )
        )

    return {
        "mean":
            float(np.mean(selected_values)),

        "median":
            float(np.median(selected_values)),

        "maximum":
            float(np.max(selected_values)),

        "minimum":
            float(np.min(selected_values)),

        "std_dev":
            float(
                np.std(
                    selected_values,
                    ddof=1
                )
            )
            if len(selected_values) > 1
            else 0.0,

        "integral":
            integral,

        "samples":
            len(selected_values)
    }


# =========================================================
# DISCOVER FILES
# =========================================================

nc_files = sorted(
    BASE_FOLDER.rglob("*.nc")
)

shot_files = {}


for file_path in nc_files:

    diagnostic, shot = (
        normalise_filename(
            file_path.name
        )
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
# PROCESS EACH SHOT
# =========================================================

segment_rows = []


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
            "  Missing core diagnostic"
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
        signal is None
        for signal in [
            plasma,
            total_nbi,
            ss_nbi,
            sw_nbi,
            neutrons
        ]
    ):

        print(
            "  Required signal missing"
        )

        continue


    # -----------------------------------------------------
    # USE TOTAL NBI TIMEBASE
    # -----------------------------------------------------

    nbi_time = total_nbi["time"]

    total_values = (
        total_nbi["values"]
    )

    ss_values = interpolate_to_time(
        ss_nbi["time"],
        ss_nbi["values"],
        nbi_time
    )

    sw_values = interpolate_to_time(
        sw_nbi["time"],
        sw_nbi["values"],
        nbi_time
    )


    # -----------------------------------------------------
    # INITIAL STATES
    # -----------------------------------------------------

    states = build_initial_state_series(
        nbi_time,
        total_values,
        ss_values,
        sw_values
    )


    segments = find_contiguous_segments(
        nbi_time,
        states
    )


    # Remove tiny switching artefacts
    segments = merge_short_segments(
        nbi_time,
        segments
    )


    # Split sustained state if power changes materially
    segments = split_on_large_power_change(
        nbi_time,
        total_values,
        segments
    )


    # -----------------------------------------------------
    # ONLY KEEP NBI STATES
    # -----------------------------------------------------

    nbi_segments = [
        segment
        for segment in segments
        if segment["state"] != "No NBI"
    ]


    # -----------------------------------------------------
    # GENERATE METRICS
    # -----------------------------------------------------

    for segment_number, segment in enumerate(
        nbi_segments,
        start=1
    ):

        start_index = segment[
            "start_index"
        ]

        end_index = segment[
            "end_index"
        ]

        start_time = float(
            nbi_time[start_index]
        )

        end_time = float(
            nbi_time[end_index]
        )

        duration = float(
            end_time
            - start_time
        )


        # NBI
        total_metrics = (
            calculate_metrics(
                nbi_time,
                total_values,
                start_time,
                end_time
            )
        )


        ss_metrics = (
            calculate_metrics(
                nbi_time,
                ss_values,
                start_time,
                end_time
            )
        )


        sw_metrics = (
            calculate_metrics(
                nbi_time,
                sw_values,
                start_time,
                end_time
            )
        )


        # Plasma current during same interval
        plasma_metrics = (
            calculate_metrics(
                plasma["time"],
                plasma["values"],
                start_time,
                end_time
            )
        )


        # Neutron response during same interval
        neutron_metrics = (
            calculate_metrics(
                neutrons["time"],
                neutrons["values"],
                start_time,
                end_time
            )
        )


        # Beam mode based on actual segment averages
        ss_active = (
            ss_metrics["mean"]
            > BEAM_ON_THRESHOLD_MW
        )

        sw_active = (
            sw_metrics["mean"]
            > BEAM_ON_THRESHOLD_MW
        )


        if ss_active and sw_active:

            beam_mode = "SS + SW"

        elif ss_active:

            beam_mode = "SS only"

        elif sw_active:

            beam_mode = "SW only"

        else:

            beam_mode = "No NBI"


        segment_rows.append({

            "shot":
                shot,

            "segment_number":
                segment_number,

            "beam_mode":
                beam_mode,

            "segment_start_s":
                start_time,

            "segment_end_s":
                end_time,

            "segment_duration_s":
                duration,

            "mean_total_nbi_MW":
                total_metrics["mean"],

            "median_total_nbi_MW":
                total_metrics["median"],

            "peak_total_nbi_MW":
                total_metrics["maximum"],

            "mean_ss_nbi_MW":
                ss_metrics["mean"],

            "mean_sw_nbi_MW":
                sw_metrics["mean"],

            "mean_plasma_current_kA":
                plasma_metrics["mean"],

            "peak_plasma_current_kA":
                plasma_metrics["maximum"],

            "mean_neutron_rate_n_s":
                neutron_metrics["mean"],

            "peak_neutron_rate_n_s":
                neutron_metrics["maximum"],

            "integrated_neutron_output":
                neutron_metrics["integral"],

            "nbi_energy_proxy_MJ":
                total_metrics["integral"]

        })


    # =====================================================
    # QA PLOT
    # =====================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True
    )


    # NBI power
    axes[0].plot(
        total_nbi["time"],
        total_nbi["values"],
        label="Total NBI"
    )

    axes[0].plot(
        ss_nbi["time"],
        ss_nbi["values"],
        alpha=0.6,
        label="SS"
    )

    axes[0].plot(
        sw_nbi["time"],
        sw_nbi["values"],
        alpha=0.6,
        label="SW"
    )

    axes[0].axhline(
        BEAM_ON_THRESHOLD_MW,
        linestyle="--",
        alpha=0.6
    )


    # Neutrons
    axes[1].plot(
        neutrons["time"],
        neutrons["values"]
    )


    # Segment boundaries
    for segment in nbi_segments:

        start_time = nbi_time[
            segment["start_index"]
        ]

        end_time = nbi_time[
            segment["end_index"]
        ]

        for axis in axes:

            axis.axvline(
                start_time,
                linestyle="--",
                alpha=0.7
            )

            axis.axvline(
                end_time,
                linestyle="--",
                alpha=0.7
            )


    axes[0].set_ylabel(
        "NBI Power (MW)"
    )

    axes[1].set_ylabel(
        "Neutron Rate (n/s)"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )


    axes[0].legend()


    fig.suptitle(
        f"Shot {shot} — "
        f"Detected NBI Power States",
        fontsize=14,
        fontweight="bold"
    )


    for axis in axes:

        axis.grid(
            alpha=0.25
        )

        axis.set_xlim(
            -0.05,
            0.60
        )


    fig.tight_layout()


    fig.savefig(
        QA_FOLDER
        / f"shot_{shot}_nbi_states.png",
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# =========================================================
# CREATE OUTPUT TABLE
# =========================================================

segments_df = pd.DataFrame(
    segment_rows
)


if not segments_df.empty:

    segments_df = (
        segments_df.sort_values(
            [
                "shot",
                "segment_number"
            ]
        )
    )


segments_df.to_csv(
    OUTPUT_FOLDER
    / "nbi_state_segments_v1.csv",
    index=False
)


# =========================================================
# SHOT-LEVEL SEGMENT SUMMARY
# =========================================================

if not segments_df.empty:

    shot_summary = (

        segments_df
        .groupby("shot")
        .agg(

            nbi_segments=(
                "segment_number",
                "count"
            ),

            beam_modes=(
                "beam_mode",
                lambda x:
                " | ".join(
                    sorted(
                        set(x)
                    )
                )
            ),

            max_nbi_power_MW=(
                "peak_total_nbi_MW",
                "max"
            ),

            max_neutron_rate_n_s=(
                "peak_neutron_rate_n_s",
                "max"
            ),

            total_integrated_neutrons=(
                "integrated_neutron_output",
                "sum"
            )
        )

        .reset_index()
    )

else:

    shot_summary = pd.DataFrame()


shot_summary.to_csv(
    OUTPUT_FOLDER
    / "nbi_state_shot_summary.csv",
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 60)
print(
    "V4 NBI STATE SEGMENTATION COMPLETE"
)
print("=" * 60)

print(
    f"Shots analysed: "
    f"{len(shot_files)}"
)

print(
    f"NBI segments detected: "
    f"{len(segments_df)}"
)

print()

if not segments_df.empty:

    print(
        segments_df[
            [
                "shot",
                "segment_number",
                "beam_mode",
                "segment_start_s",
                "segment_end_s",
                "mean_total_nbi_MW",
                "mean_neutron_rate_n_s"
            ]
        ].to_string(
            index=False
        )
    )

print()
print("Outputs:")
print(
    OUTPUT_FOLDER
    / "nbi_state_segments_v1.csv"
)

print(
    OUTPUT_FOLDER
    / "nbi_state_shot_summary.csv"
)

print(
    QA_FOLDER
)
