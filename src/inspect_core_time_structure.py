from pathlib import Path
import re

import h5py


BASE_FOLDER = Path(__file__).parent


TARGET_SIGNALS = {
    "AMC": "AMC_PLASMA CURRENT",
    "ANB": "ANB_TOT_SUM_POWER",
    "ANU": "ANU_NEUTRONS"
}


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


def clean(value):

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace"
        )

    if hasattr(value, "tolist"):
        return value.tolist()

    return value


# Only inspect one representative shot
SHOT_TO_INSPECT = 27929


files = sorted(
    BASE_FOLDER.rglob("*.nc")
)


for file_path in files:

    diagnostic, shot = normalise_filename(
        file_path.name
    )

    if shot != SHOT_TO_INSPECT:
        continue

    if diagnostic not in TARGET_SIGNALS:
        continue

    target_signal = TARGET_SIGNALS[
        diagnostic
    ]

    print()
    print("=" * 70)
    print(
        f"SHOT {shot} | "
        f"{diagnostic} | "
        f"{file_path.name}"
    )
    print("=" * 70)

    with h5py.File(
        file_path,
        "r"
    ) as hdf:

        # ---------------------------------
        # PRINT COMPLETE FILE STRUCTURE
        # ---------------------------------

        print()
        print("FILE STRUCTURE")
        print("-" * 70)

        def structure_visitor(name, obj):

            if isinstance(
                obj,
                h5py.Dataset
            ):

                print(
                    f"DATASET: {name}"
                )

                print(
                    f"    shape: {obj.shape}"
                )

                print(
                    f"    dtype: {obj.dtype}"
                )

            elif isinstance(
                obj,
                h5py.Group
            ):

                print(
                    f"GROUP:   {name}"
                )

        hdf.visititems(
            structure_visitor
        )

        # ---------------------------------
        # FIND TARGET SIGNAL
        # ---------------------------------

        print()
        print("TARGET SIGNAL")
        print("-" * 70)

        def signal_visitor(name, obj):

            if not isinstance(
                obj,
                h5py.Dataset
            ):
                return

            attrs = {
                key: clean(value)
                for key, value
                in obj.attrs.items()
            }

            original_name = attrs.get(
                "original_signal_name",
                ""
            )

            if original_name != target_signal:
                return

            print()
            print(
                f"FOUND: {original_name}"
            )

            print(
                f"PATH:  {name}"
            )

            print(
                f"SHAPE: {obj.shape}"
            )

            print()
            print("ATTRIBUTES:")

            for key, value in attrs.items():

                print(
                    f"    {key}: {value}"
                )

            # ---------------------------------
            # PARENT GROUP
            # ---------------------------------

            parent_path = (
                name.rsplit("/", 1)[0]
            )

            parent = hdf[parent_path]

            print()
            print(
                f"PARENT GROUP: "
                f"{parent_path}"
            )

            print()

            print(
                "Objects inside parent:"
            )

            for child_name in parent.keys():

                child = parent[
                    child_name
                ]

                if isinstance(
                    child,
                    h5py.Dataset
                ):

                    print(
                        f"    DATASET "
                        f"{child_name}"
                        f" | shape={child.shape}"
                    )

                else:

                    print(
                        f"    GROUP "
                        f"{child_name}"
                    )

        hdf.visititems(
            signal_visitor
        )


print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
