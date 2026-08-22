from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_FOLDER = Path(__file__).parent
OUTPUT_FOLDER = BASE_FOLDER / "outputs"

DIAGNOSTIC_INPUT = OUTPUT_FOLDER / "diagnostic_catalogue.csv"
SIGNAL_INPUT = OUTPUT_FOLDER / "signal_catalogue.csv"


# ---------------------------------------------------------
# DIAGNOSTIC CLASSIFICATION
# ---------------------------------------------------------
#
# Primary category:
#     Main scientific purpose of the diagnostic.
#
# Secondary category:
#     Additional scientific area where the diagnostic may
#     provide useful information.
#
# Project role:
#     CORE       = central to first NBI/neutron analysis
#     CANDIDATE  = likely useful for expanding first analysis
#     FUTURE     = retained for later analysis
#
# These roles can change later. Scientific classification
# should remain relatively stable.
# ---------------------------------------------------------

DIAGNOSTIC_CLASSIFICATION = {

    "ABM": {
        "primary_category": "Radiation & Bolometry",
        "secondary_category": "",
        "project_role": "FUTURE"
    },

    "ACD": {
        "primary_category": "Spectroscopy & Impurity Physics",
        "secondary_category": "Ion Physics",
        "project_role": "FUTURE"
    },

    "ACT": {
        "primary_category": "Ion Temperature, Rotation & Spectroscopy",
        "secondary_category": "Plasma Transport",
        "project_role": "FUTURE"
    },

    "ADA": {
        "primary_category": "Edge & Divertor Physics",
        "secondary_category": "Plasma Geometry",
        "project_role": "FUTURE"
    },

    "ADG": {
        "primary_category": "Edge & Divertor Physics",
        "secondary_category": "Electron Density",
        "project_role": "CANDIDATE"
    },

    "AGA": {
        "primary_category": "Fuelling & Gas Injection",
        "secondary_category": "Machine Operations",
        "project_role": "FUTURE"
    },

    "AHX": {
        "primary_category": "Radiation & Energetic Particle Emission",
        "secondary_category": "Fast Ions & Energetic Particles",
        "project_role": "CANDIDATE"
    },

    "AIM": {
        "primary_category": "Edge & Divertor Physics",
        "secondary_category": "Spectroscopy & Plasma Emission",
        "project_role": "FUTURE"
    },

    "ALP": {
        "primary_category": "Edge & Divertor Physics",
        "secondary_category": "Plasma-Wall Interaction",
        "project_role": "FUTURE"
    },

    "AMA": {
        "primary_category": "MHD & Plasma Stability",
        "secondary_category": "Magnetics & Plasma Current",
        "project_role": "CANDIDATE"
    },

    "AMB": {
        "primary_category": "Magnetics & Plasma Current",
        "secondary_category": "Plasma Control",
        "project_role": "CANDIDATE"
    },

    "AMC": {
        "primary_category": "Magnetics & Plasma Current",
        "secondary_category": "Plasma Control",
        "project_role": "CORE"
    },

    "AMH": {
        "primary_category": "Magnetics & Plasma Current",
        "secondary_category": "MHD & Plasma Stability",
        "project_role": "CANDIDATE"
    },

    "AMM": {
        "primary_category": "Magnetics & Plasma Current",
        "secondary_category": "MHD & Plasma Stability",
        "project_role": "CANDIDATE"
    },

    "AMS": {
        "primary_category": "Equilibrium & Current Profile",
        "secondary_category": "Magnetics & Plasma Current",
        "project_role": "CANDIDATE"
    },

    "ANB": {
        "primary_category": "Heating & Current Drive",
        "secondary_category": "Fast Ions & Energetic Particles",
        "project_role": "CORE"
    },

    "ANE": {
        "primary_category": "Electron Density",
        "secondary_category": "Plasma State",
        "project_role": "CANDIDATE"
    },

    "ANT": {
        "primary_category": "Neutrons & Fusion Performance",
        "secondary_category": "Radiation Monitoring",
        "project_role": "CANDIDATE"
    },

    "ANU": {
        "primary_category": "Neutrons & Fusion Performance",
        "secondary_category": "Fusion Output",
        "project_role": "CORE"
    },

    "ASM": {
        "primary_category": "MHD & Plasma Stability",
        "secondary_category": "Confinement Regime",
        "project_role": "CANDIDATE"
    },

    "ASX": {
        "primary_category": "Radiation & Plasma Emission",
        "secondary_category": "MHD & Plasma Stability",
        "project_role": "CANDIDATE"
    },

    "AYC": {
        "primary_category": "Electron Density & Temperature",
        "secondary_category": "Core Plasma Profiles",
        "project_role": "CANDIDATE"
    },

    "AYE": {
        "primary_category": "Electron Density & Temperature",
        "secondary_category": "Edge Plasma Profiles",
        "project_role": "CANDIDATE"
    },

    "EFM": {
        "primary_category": "Equilibrium & Plasma Geometry",
        "secondary_category": "Plasma State",
        "project_role": "CANDIDATE"
    },
}


# ---------------------------------------------------------
# LOAD V1 OUTPUTS
# ---------------------------------------------------------

print("Loading V1 catalogues...")

diagnostics_df = pd.read_csv(DIAGNOSTIC_INPUT)
signals_df = pd.read_csv(SIGNAL_INPUT)


# ---------------------------------------------------------
# CLASSIFICATION FUNCTION
# ---------------------------------------------------------

def classify_diagnostic(code):

    code = str(code).upper()

    classification = DIAGNOSTIC_CLASSIFICATION.get(code)

    if classification is None:

        return pd.Series({
            "primary_category": "Other / Unclassified",
            "secondary_category": "",
            "project_role": "REVIEW"
        })

    return pd.Series(classification)


# ---------------------------------------------------------
# CLASSIFY DIAGNOSTIC CATALOGUE
# ---------------------------------------------------------

classification_columns = diagnostics_df["diagnostic"].apply(
    classify_diagnostic
)

diagnostics_classified = pd.concat(
    [diagnostics_df, classification_columns],
    axis=1
)


# ---------------------------------------------------------
# CLASSIFY SIGNAL CATALOGUE
# ---------------------------------------------------------

signal_classification = signals_df["diagnostic"].apply(
    classify_diagnostic
)

signals_classified = pd.concat(
    [signals_df, signal_classification],
    axis=1
)


# ---------------------------------------------------------
# ADD COVERAGE BAND
# ---------------------------------------------------------

def coverage_band(value):

    if value == 100:
        return "Complete"

    elif value >= 80:
        return "High"

    elif value >= 50:
        return "Moderate"

    elif value > 0:
        return "Low"

    return "None"


diagnostics_classified["coverage_band"] = (
    diagnostics_classified["coverage_pct"]
    .apply(coverage_band)
)

signals_classified["coverage_band"] = (
    signals_classified["coverage_pct"]
    .apply(coverage_band)
)


# ---------------------------------------------------------
# ADD ANALYSIS PRIORITY
# ---------------------------------------------------------
#
# This is deliberately separate from project role.
#
# Priority combines:
#     project relevance
#     +
#     availability across the cohort
# ---------------------------------------------------------

def analysis_priority(row):

    role = row["project_role"]
    coverage = row["coverage_pct"]

    if role == "CORE" and coverage >= 80:
        return "1 - High"

    if role == "CANDIDATE" and coverage >= 80:
        return "2 - Medium"

    if role == "CANDIDATE":
        return "3 - Review"

    if role == "FUTURE":
        return "4 - Future"

    return "5 - Unclassified"


diagnostics_classified["analysis_priority"] = (
    diagnostics_classified.apply(
        analysis_priority,
        axis=1
    )
)

signals_classified["analysis_priority"] = (
    signals_classified.apply(
        analysis_priority,
        axis=1
    )
)


# ---------------------------------------------------------
# IDENTIFY UNCLASSIFIED DIAGNOSTICS
# ---------------------------------------------------------

unclassified_df = diagnostics_classified[
    diagnostics_classified["primary_category"]
    == "Other / Unclassified"
].copy()


# ---------------------------------------------------------
# CREATE CATEGORY SUMMARY
# ---------------------------------------------------------

category_summary = (
    diagnostics_classified
    .groupby("primary_category")
    .agg(
        diagnostics=("diagnostic", "nunique"),
        mean_coverage_pct=("coverage_pct", "mean")
    )
    .reset_index()
)

category_summary["mean_coverage_pct"] = (
    category_summary["mean_coverage_pct"]
    .round(1)
)


# ---------------------------------------------------------
# CREATE PROJECT ROLE SUMMARY
# ---------------------------------------------------------

role_summary = (
    diagnostics_classified
    .groupby("project_role")
    .agg(
        diagnostics=("diagnostic", "nunique"),
        mean_coverage_pct=("coverage_pct", "mean")
    )
    .reset_index()
)

role_summary["mean_coverage_pct"] = (
    role_summary["mean_coverage_pct"]
    .round(1)
)


# ---------------------------------------------------------
# SORT OUTPUTS
# ---------------------------------------------------------

diagnostics_classified = diagnostics_classified.sort_values(
    [
        "analysis_priority",
        "primary_category",
        "diagnostic"
    ]
)

signals_classified = signals_classified.sort_values(
    [
        "analysis_priority",
        "primary_category",
        "diagnostic",
        "signal_name"
    ]
)


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

diagnostics_classified.to_csv(
    OUTPUT_FOLDER / "diagnostic_catalogue_classified.csv",
    index=False
)

signals_classified.to_csv(
    OUTPUT_FOLDER / "signal_catalogue_classified.csv",
    index=False
)

category_summary.to_csv(
    OUTPUT_FOLDER / "scientific_category_summary.csv",
    index=False
)

role_summary.to_csv(
    OUTPUT_FOLDER / "project_role_summary.csv",
    index=False
)

unclassified_df.to_csv(
    OUTPUT_FOLDER / "diagnostics_requiring_review.csv",
    index=False
)


# ---------------------------------------------------------
# CONSOLE SUMMARY
# ---------------------------------------------------------

print()
print("----------------------------------------")
print("V2 CLASSIFICATION COMPLETE")
print("----------------------------------------")

print(
    f"Diagnostics classified: "
    f"{len(diagnostics_classified)}"
)

print(
    f"Signals classified:     "
    f"{len(signals_classified)}"
)

print(
    f"Unclassified:           "
    f"{len(unclassified_df)}"
)

print()
print("Project roles:")
print()

print(
    role_summary.to_string(index=False)
)

print()
print("Scientific categories:")
print()

print(
    category_summary.to_string(index=False)
)

print()
print("Outputs saved to:")
print(OUTPUT_FOLDER.resolve())
