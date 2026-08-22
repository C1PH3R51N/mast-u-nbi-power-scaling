from pathlib import Path
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

BASE_FOLDER = Path(__file__).parent

PLOT_FOLDER = (
    BASE_FOLDER
    / "outputs"
    / "core_signal_plots"
)

PLOT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


TARGET_SIGNALS = {

    "AMC": "AMC_PLASMA CURRENT",

    "ANB": "ANB_TOT_SUM_POWER",

    "ANU": "ANU_NEUTRONS"
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


def find_dimension_coordinate(
    hdf,
    dataset
):
    """
    Use the HDF5 DIMENSION_LIST reference to locate
    the correct coordinate/time dataset.
    """

    try:

        dimension_list = (
            dataset.attrs.get(
                "DIMENSION_LIST"
            )
        )

        if dimension_list is None:
            return None

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
                        == dataset.shape[0]
                    ):

                        return np.asarray(
                            coordinate[()],
                            dtype=float
                        ).ravel()

                except Exception:
                    pass

    except Exception:
        pass

    return None


def extract_target_signal(
    file_path,
    target_signal
):
    """
    Find a target signal and return:

        time
        values
        units
        label
    """

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

            time = find_dimension_coordinate(
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
                    target_signal
                )
            }

        hdf.visititems(
            visitor
        )

    return result


# =========================================================
# FIND CORE FILES
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

    if shot not in shot_files:
        shot_files[shot] = {}

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
# CREATE ONE FIGURE PER SHOT
# =========================================================

plots_created = 0


for shot in sorted(
    shot_files.keys()
):

    print(
        f"Creating plot for "
        f"shot {shot}..."
    )


    shot_data = {}


    # -----------------------------------------------------
    # LOAD THE THREE CORE SIGNALS
    # -----------------------------------------------------

    for diagnostic, signal_name in (
        TARGET_SIGNALS.items()
    ):

        file_path = (
            shot_files[
                shot
            ].get(
                diagnostic
            )
        )

        if file_path is None:

            print(
                f"  Missing {diagnostic}"
            )

            continue


        data = extract_target_signal(
            file_path,
            signal_name
        )


        if data is None:

            print(
                f"  Signal not found: "
                f"{signal_name}"
            )

            continue


        if data["time"] is None:

            print(
                f"  No time coordinate: "
                f"{signal_name}"
            )

            continue


        shot_data[
            diagnostic
        ] = data


    # -----------------------------------------------------
    # REQUIRE ALL THREE
    # -----------------------------------------------------

    if len(shot_data) != 3:

        print(
            f"  Shot {shot} skipped - "
            f"incomplete core data."
        )

        continue


    # -----------------------------------------------------
    # CREATE FIGURE
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )


    # -----------------------------------------------------
    # PLASMA CURRENT
    # -----------------------------------------------------

    amc = shot_data["AMC"]

    axes[0].plot(
        amc["time"],
        amc["values"]
    )

    axes[0].set_ylabel(
        "Plasma Current\n(kA)"
    )

    axes[0].set_title(
        f"MAST Shot {shot} — "
        f"Core Signal Overview",
        fontsize=15,
        fontweight="bold"
    )

    axes[0].grid(
        alpha=0.25
    )


    # -----------------------------------------------------
    # NBI POWER
    # -----------------------------------------------------

    anb = shot_data["ANB"]

    axes[1].plot(
        anb["time"],
        anb["values"]
    )

    axes[1].set_ylabel(
        "Total NBI Power\n(MW)"
    )

    axes[1].grid(
        alpha=0.25
    )


    # -----------------------------------------------------
    # NEUTRON RATE
    # -----------------------------------------------------

    anu = shot_data["ANU"]

    axes[2].plot(
        anu["time"],
        anu["values"]
    )

    axes[2].set_ylabel(
        "Neutron Rate\n(n/s)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].grid(
        alpha=0.25
    )


    # -----------------------------------------------------
    # COMMON DISPLAY WINDOW
    # -----------------------------------------------------
    #
    # This does NOT remove data.
    #
    # It simply gives every figure the same visual
    # x-axis so that shots can be compared consistently.
    # -----------------------------------------------------

    axes[2].set_xlim(
        -0.25,
        1.25
    )


    # -----------------------------------------------------
    # ZERO REFERENCE
    # -----------------------------------------------------

    for axis in axes:

        axis.axhline(
            0,
            linewidth=0.8,
            alpha=0.4
        )

        axis.axvline(
            0,
            linewidth=0.8,
            linestyle="--",
            alpha=0.4
        )


    # -----------------------------------------------------
    # FINISH FIGURE
    # -----------------------------------------------------

    fig.tight_layout()


    output_path = (
        PLOT_FOLDER
        / f"shot_{shot}_core_signals.png"
    )


    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    plots_created += 1


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 60)
print(
    "V3B CORE SIGNAL PLOTS COMPLETE"
)
print("=" * 60)

print(
    f"Plots created: "
    f"{plots_created}"
)

print()
print(
    "Saved to:"
)

print(
    PLOT_FOLDER.resolve()
)