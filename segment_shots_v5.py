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

QA_FOLDER = OUTPUT_FOLDER / "nbi_state_qa_v5"
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
            duration_before >= MIN_STATE_DURATION_S
            and duration_after >= MIN_STATE_DURATION_S
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
        "mean": float(np.mean(selected_values)),
        "median": float(np.median(selected_values)),
        "maximum": float(np.max(selected_values)),
        "minimum": float(np.min(selected_values)),
        "std_dev": (
            float(
                np.std(
                    selected_values,
                    ddof=1
                )
            )
            if len(selected_values) > 1
            else 0.0
        ),
        "integral": integral,
        "samples": len(selected_values)
    }


# =========================================================
# V5 RESPONSE METRICS
# =========================================================

def sample_at_time(time, values, target_time):
    """
    Linearly interpolate a signal value at one time.
    """

    mask = (
        np.isfinite(time)
        & np.isfinite(values)
    )

    t = np.asarray(time)[mask]
    v = np.asarray(values)[mask]

    if (
        len(t) < 2
        or target_time < t[0]
        or target_time > t[-1]
    ):
        return np.nan

    return float(
        np.interp(
            target_time,
            t,
            v
        )
    )


def neutron_response_metrics(
    time,
    values,
    start_time,
    end_time
):
    """
    Characterise neutron response within one detected NBI state.

    The neutron integral has units of neutrons because
    neutron rate (n/s) is integrated over seconds.
    """

    mask = (
        (time >= start_time)
        & (time <= end_time)
        & np.isfinite(time)
        & np.isfinite(values)
    )

    t = np.asarray(time)[mask]
    y = np.asarray(values)[mask]

    empty = {
        "entry": np.nan,
        "exit": np.nan,
        "mean": np.nan,
        "peak": np.nan,
        "peak_time": np.nan,
        "time_to_peak": np.nan,
        "integral": np.nan,
        "net_change": np.nan,
        "overall_slope": np.nan,
        "early_rise_rate": np.nan,
        "samples": 0
    }

    if len(y) == 0:
        return empty

    peak_i = int(
        np.nanargmax(y)
    )

    peak_time = float(
        t[peak_i]
    )

    entry = float(
        y[0]
    )

    exit_value = float(
        y[-1]
    )

    duration = (
        float(t[-1] - t[0])
        if len(t) >= 2
        else 0.0
    )

    integral = (
        float(
            np.trapezoid(
                y,
                t
            )
        )
        if len(t) >= 2
        else np.nan
    )

    overall_slope = (
        float(
            (exit_value - entry)
            / duration
        )
        if duration > 0
        else np.nan
    )

    # First 30 ms after the NBI state begins
    early_end = min(
        start_time + 0.030,
        end_time
    )

    early_mask = (
        (t >= start_time)
        & (t <= early_end)
    )

    early_rise_rate = np.nan

    if np.count_nonzero(
        early_mask
    ) >= 2:

        te = t[early_mask]
        ye = y[early_mask]

        finite = (
            np.isfinite(te)
            & np.isfinite(ye)
        )

        if np.count_nonzero(
            finite
        ) >= 2:

            early_rise_rate = float(
                np.polyfit(
                    te[finite],
                    ye[finite],
                    1
                )[0]
            )

    return {
        "entry": entry,
        "exit": exit_value,
        "mean": float(np.mean(y)),
        "peak": float(y[peak_i]),
        "peak_time": peak_time,
        "time_to_peak": float(
            peak_time - start_time
        ),
        "integral": integral,
        "net_change": float(
            exit_value - entry
        ),
        "overall_slope": overall_slope,
        "early_rise_rate": early_rise_rate,
        "samples": int(len(y))
    }


def preceding_window_metrics(
    time,
    values,
    start_time,
    window_s=0.020
):
    """
    Mean signal in the window immediately before
    a state begins.
    """

    return calculate_metrics(
        time,
        values,
        start_time - window_s,
        start_time
    )


def safe_delta(
    current,
    previous
):

    if not (
        np.isfinite(current)
        and np.isfinite(previous)
    ):
        return np.nan

    return float(
        current - previous
    )


def safe_ratio(
    current,
    previous
):

    if not (
        np.isfinite(current)
        and np.isfinite(previous)
        and previous != 0
    ):
        return np.nan

    return float(
        current / previous
    )
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
    # LOAD CORE SIGNALS
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
    # USE TOTAL NBI TIMEBASE FOR BEAM STATE DETECTION
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
    # BUILD INITIAL NBI STATES
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


    # Remove short switching artefacts
    segments = merge_short_segments(
        nbi_time,
        segments
    )


    # Split sustained beam period if total power
    # changes materially within the state
    segments = split_on_large_power_change(
        nbi_time,
        total_values,
        segments
    )


    # -----------------------------------------------------
    # KEEP ONLY REAL NBI STATES
    # -----------------------------------------------------

    nbi_segments = [
        segment
        for segment in segments
        if segment["state"] != "No NBI"
    ]


    # -----------------------------------------------------
    # STATE-BY-STATE ANALYSIS
    # -----------------------------------------------------

    previous_state_row = None


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


        # -------------------------------------------------
        # NBI METRICS
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PLASMA CURRENT DURING THIS NBI STATE
        # -------------------------------------------------

        plasma_metrics = (
            calculate_metrics(
                plasma["time"],
                plasma["values"],
                start_time,
                end_time
            )
        )


        plasma_entry = (
            sample_at_time(
                plasma["time"],
                plasma["values"],
                start_time
            )
        )


        pre_plasma_metrics = (
            preceding_window_metrics(
                plasma["time"],
                plasma["values"],
                start_time,
                window_s=0.020
            )
        )


        # -------------------------------------------------
        # NEUTRON RESPONSE
        # -------------------------------------------------

        neutron_metrics = (
            calculate_metrics(
                neutrons["time"],
                neutrons["values"],
                start_time,
                end_time
            )
        )


        neutron_response = (
            neutron_response_metrics(
                neutrons["time"],
                neutrons["values"],
                start_time,
                end_time
            )
        )


        neutron_entry = (
            sample_at_time(
                neutrons["time"],
                neutrons["values"],
                start_time
            )
        )


        pre_neutron_metrics = (
            preceding_window_metrics(
                neutrons["time"],
                neutrons["values"],
                start_time,
                window_s=0.020
            )
        )


        # -------------------------------------------------
        # BEAM MODE FOR THIS STATE
        # -------------------------------------------------

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


        # -------------------------------------------------
        # CREATE STATE ROW
        # -------------------------------------------------

        row = {

            # ---------------------------------------------
            # IDENTIFICATION
            # ---------------------------------------------

            "shot":
                shot,

            "segment_number":
                segment_number,

            "beam_mode":
                beam_mode,


            # ---------------------------------------------
            # TIMING
            # ---------------------------------------------

            "segment_start_s":
                start_time,

            "segment_end_s":
                end_time,

            "segment_duration_s":
                duration,


            # ---------------------------------------------
            # TOTAL NBI
            # ---------------------------------------------

            "mean_total_nbi_MW":
                total_metrics["mean"],

            "median_total_nbi_MW":
                total_metrics["median"],

            "peak_total_nbi_MW":
                total_metrics["maximum"],

            "std_total_nbi_MW":
                total_metrics["std_dev"],

            # MW integrated over seconds = MJ
            "nbi_energy_MJ":
                total_metrics["integral"],


            # ---------------------------------------------
            # SS BEAM
            # ---------------------------------------------

            "mean_ss_nbi_MW":
                ss_metrics["mean"],

            "peak_ss_nbi_MW":
                ss_metrics["maximum"],

            "std_ss_nbi_MW":
                ss_metrics["std_dev"],


            # ---------------------------------------------
            # SW BEAM
            # ---------------------------------------------

            "mean_sw_nbi_MW":
                sw_metrics["mean"],

            "peak_sw_nbi_MW":
                sw_metrics["maximum"],

            "std_sw_nbi_MW":
                sw_metrics["std_dev"],


            # ---------------------------------------------
            # PLASMA CURRENT
            # ---------------------------------------------

            "plasma_current_at_state_entry_kA":
                plasma_entry,

            "mean_plasma_current_kA":
                plasma_metrics["mean"],

            "peak_plasma_current_kA":
                plasma_metrics["maximum"],

            "std_plasma_current_kA":
                plasma_metrics["std_dev"],

            "pre_state_20ms_mean_plasma_current_kA":
                pre_plasma_metrics["mean"],


            # ---------------------------------------------
            # NEUTRON RESPONSE
            # ---------------------------------------------

            "neutron_rate_at_state_entry_n_s":
                neutron_entry,

            "neutron_rate_at_state_exit_n_s":
                neutron_response["exit"],

            "mean_neutron_rate_n_s":
                neutron_metrics["mean"],

            "peak_neutron_rate_n_s":
                neutron_metrics["maximum"],

            "std_neutron_rate_n_s":
                neutron_metrics["std_dev"],

            "neutron_peak_time_s":
                neutron_response["peak_time"],

            "neutron_time_to_peak_s":
                neutron_response["time_to_peak"],

            "integrated_neutron_output":
                neutron_response["integral"],

            "neutron_net_change_n_s":
                neutron_response["net_change"],

            "neutron_overall_slope_n_s2":
                neutron_response["overall_slope"],

            "neutron_early_30ms_rise_rate_n_s2":
                neutron_response["early_rise_rate"],


            # ---------------------------------------------
            # PRE-STATE BASELINE
            # ---------------------------------------------

            "pre_state_20ms_mean_neutron_rate_n_s":
                pre_neutron_metrics["mean"],

            "delta_mean_neutron_vs_pre20ms_n_s":
                safe_delta(
                    neutron_metrics["mean"],
                    pre_neutron_metrics["mean"]
                ),

            "neutron_gain_vs_pre20ms":
                safe_ratio(
                    neutron_metrics["mean"],
                    pre_neutron_metrics["mean"]
                ),
        }


        # -------------------------------------------------
        # CHANGE RELATIVE TO PREVIOUS NBI STATE
        # -------------------------------------------------

        if previous_state_row is None:

            row[
                "previous_beam_mode"
            ] = ""

            row[
                "delta_total_nbi_vs_previous_MW"
            ] = np.nan

            row[
                "delta_mean_neutron_vs_previous_n_s"
            ] = np.nan

            row[
                "neutron_ratio_vs_previous"
            ] = np.nan

            row[
                "delta_integrated_neutrons_vs_previous"
            ] = np.nan

            row[
                "delta_mean_plasma_current_vs_previous_kA"
            ] = np.nan


        else:

            row[
                "previous_beam_mode"
            ] = previous_state_row[
                "beam_mode"
            ]


            row[
                "delta_total_nbi_vs_previous_MW"
            ] = safe_delta(
                row[
                    "mean_total_nbi_MW"
                ],
                previous_state_row[
                    "mean_total_nbi_MW"
                ]
            )


            row[
                "delta_mean_neutron_vs_previous_n_s"
            ] = safe_delta(
                row[
                    "mean_neutron_rate_n_s"
                ],
                previous_state_row[
                    "mean_neutron_rate_n_s"
                ]
            )


            row[
                "neutron_ratio_vs_previous"
            ] = safe_ratio(
                row[
                    "mean_neutron_rate_n_s"
                ],
                previous_state_row[
                    "mean_neutron_rate_n_s"
                ]
            )


            row[
                "delta_integrated_neutrons_vs_previous"
            ] = safe_delta(
                row[
                    "integrated_neutron_output"
                ],
                previous_state_row[
                    "integrated_neutron_output"
                ]
            )


            row[
                "delta_mean_plasma_current_vs_previous_kA"
            ] = safe_delta(
                row[
                    "mean_plasma_current_kA"
                ],
                previous_state_row[
                    "mean_plasma_current_kA"
                ]
            )


        segment_rows.append(
            row
        )

        previous_state_row = row


    # =====================================================
    # V5 QA PLOT
    # =====================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )


    # -----------------------------------------------------
    # NBI POWER
    # -----------------------------------------------------

    axes[0].plot(
        total_nbi["time"],
        total_nbi["values"],
        label="Total NBI"
    )

    axes[0].plot(
        ss_nbi["time"],
        ss_nbi["values"],
        alpha=0.7,
        label="SS"
    )

    axes[0].plot(
        sw_nbi["time"],
        sw_nbi["values"],
        alpha=0.7,
        label="SW"
    )

    axes[0].axhline(
        BEAM_ON_THRESHOLD_MW,
        linestyle="--",
        alpha=0.5
    )

    axes[0].set_ylabel(
        "NBI Power\n(MW)"
    )

    axes[0].legend()


    # -----------------------------------------------------
    # PLASMA CURRENT
    # -----------------------------------------------------

    axes[1].plot(
        plasma["time"],
        plasma["values"]
    )

    axes[1].set_ylabel(
        "Plasma Current\n(kA)"
    )


    # -----------------------------------------------------
    # NEUTRON RATE
    # -----------------------------------------------------

    axes[2].plot(
        neutrons["time"],
        neutrons["values"]
    )

    axes[2].set_ylabel(
        "Neutron Rate\n(n/s)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )


    # -----------------------------------------------------
    # DRAW STATE BOUNDARIES
    # -----------------------------------------------------

    for segment_number, segment in enumerate(
        nbi_segments,
        start=1
    ):

        state_start = nbi_time[
            segment["start_index"]
        ]

        state_end = nbi_time[
            segment["end_index"]
        ]


        for axis in axes:

            axis.axvline(
                state_start,
                linestyle="--",
                alpha=0.7
            )

            axis.axvline(
                state_end,
                linestyle="--",
                alpha=0.7
            )


        # label state on NBI plot
        midpoint = (
            state_start
            + state_end
        ) / 2


        axes[0].text(
            midpoint,
            0.95,
            f"State {segment_number}",
            transform=axes[0].get_xaxis_transform(),
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=9
        )


    # -----------------------------------------------------
    # FIGURE FORMATTING
    # -----------------------------------------------------

    fig.suptitle(
        f"Shot {shot} — "
        f"V5 NBI State Response Analysis",
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
        / f"shot_{shot}_v5_response.png",
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# =========================================================
# CREATE V5 STATE RESPONSE TABLE
# =========================================================

response_df = pd.DataFrame(
    segment_rows
)


if not response_df.empty:

    response_df = (
        response_df.sort_values(
            [
                "shot",
                "segment_number"
            ]
        )
    )


response_df.to_csv(
    OUTPUT_FOLDER
    / "nbi_state_response_v5.csv",
    index=False
)


# =========================================================
# SHOT SUMMARY
# =========================================================

if not response_df.empty:

    shot_summary = (

        response_df
        .groupby(
            "shot"
        )
        .agg(

            nbi_states=(
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

            maximum_nbi_power_MW=(
                "peak_total_nbi_MW",
                "max"
            ),

            mean_nbi_power_MW=(
                "mean_total_nbi_MW",
                "mean"
            ),

            maximum_neutron_rate_n_s=(
                "peak_neutron_rate_n_s",
                "max"
            ),

            mean_neutron_rate_n_s=(
                "mean_neutron_rate_n_s",
                "mean"
            ),

            total_integrated_neutrons=(
                "integrated_neutron_output",
                "sum"
            ),

            mean_plasma_current_kA=(
                "mean_plasma_current_kA",
                "mean"
            )
        )

        .reset_index()
    )

else:

    shot_summary = pd.DataFrame()


shot_summary.to_csv(
    OUTPUT_FOLDER
    / "nbi_state_shot_summary_v5.csv",
    index=False
)


# =========================================================
# IDENTIFY WITHIN-SHOT STATE TRANSITIONS
# =========================================================

transition_rows = []


if not response_df.empty:

    for shot, shot_data in (
        response_df.groupby(
            "shot"
        )
    ):

        shot_data = (
            shot_data.sort_values(
                "segment_number"
            )
            .reset_index(
                drop=True
            )
        )


        if len(shot_data) < 2:
            continue


        for i in range(
            1,
            len(shot_data)
        ):

            previous = (
                shot_data.iloc[
                    i - 1
                ]
            )

            current = (
                shot_data.iloc[
                    i
                ]
            )


            transition_rows.append({

                "shot":
                    shot,

                "from_segment":
                    previous[
                        "segment_number"
                    ],

                "to_segment":
                    current[
                        "segment_number"
                    ],

                "from_beam_mode":
                    previous[
                        "beam_mode"
                    ],

                "to_beam_mode":
                    current[
                        "beam_mode"
                    ],

                "from_mean_nbi_MW":
                    previous[
                        "mean_total_nbi_MW"
                    ],

                "to_mean_nbi_MW":
                    current[
                        "mean_total_nbi_MW"
                    ],

                "delta_nbi_MW":
                    current[
                        "mean_total_nbi_MW"
                    ]
                    - previous[
                        "mean_total_nbi_MW"
                    ],

                "from_mean_neutron_rate_n_s":
                    previous[
                        "mean_neutron_rate_n_s"
                    ],

                "to_mean_neutron_rate_n_s":
                    current[
                        "mean_neutron_rate_n_s"
                    ],

                "delta_mean_neutron_rate_n_s":
                    current[
                        "mean_neutron_rate_n_s"
                    ]
                    - previous[
                        "mean_neutron_rate_n_s"
                    ],

                "neutron_rate_ratio":
                    safe_ratio(
                        current[
                            "mean_neutron_rate_n_s"
                        ],
                        previous[
                            "mean_neutron_rate_n_s"
                        ]
                    ),

                "from_mean_plasma_current_kA":
                    previous[
                        "mean_plasma_current_kA"
                    ],

                "to_mean_plasma_current_kA":
                    current[
                        "mean_plasma_current_kA"
                    ],

                "delta_mean_plasma_current_kA":
                    current[
                        "mean_plasma_current_kA"
                    ]
                    - previous[
                        "mean_plasma_current_kA"
                    ],

                "transition_time_s":
                    current[
                        "segment_start_s"
                    ]
            })


transition_df = pd.DataFrame(
    transition_rows
)


transition_df.to_csv(
    OUTPUT_FOLDER
    / "nbi_state_transitions_v5.csv",
    index=False
)


# =========================================================
# CONSOLE SUMMARY
# =========================================================

print()
print("=" * 65)
print(
    "V5 NBI STATE RESPONSE ANALYSIS COMPLETE"
)
print("=" * 65)

print(
    f"Shots discovered:       "
    f"{len(shot_files)}"
)

print(
    f"NBI states analysed:    "
    f"{len(response_df)}"
)

print(
    f"State transitions:      "
    f"{len(transition_df)}"
)


print()
print(
    "STATE-LEVEL RESULTS"
)
print("-" * 65)


if not response_df.empty:

    print(
        response_df[
            [
                "shot",
                "segment_number",
                "beam_mode",
                "segment_start_s",
                "segment_end_s",
                "mean_total_nbi_MW",
                "mean_plasma_current_kA",
                "mean_neutron_rate_n_s",
                "peak_neutron_rate_n_s",
                "neutron_time_to_peak_s"
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    "WITHIN-SHOT TRANSITIONS"
)
print("-" * 65)


if transition_df.empty:

    print(
        "No multi-state transitions found."
    )

else:

    print(
        transition_df[
            [
                "shot",
                "from_beam_mode",
                "to_beam_mode",
                "delta_nbi_MW",
                "delta_mean_neutron_rate_n_s",
                "neutron_rate_ratio",
                "delta_mean_plasma_current_kA"
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    "Outputs saved to:"
)

print(
    OUTPUT_FOLDER
    / "nbi_state_response_v5.csv"
)

print(
    OUTPUT_FOLDER
    / "nbi_state_shot_summary_v5.csv"
)

print(
    OUTPUT_FOLDER
    / "nbi_state_transitions_v5.csv"
)

print(
    QA_FOLDER
)