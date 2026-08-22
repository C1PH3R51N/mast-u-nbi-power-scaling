# MAST-U NBI Power Scaling Analysis

An end-to-end data analysis project investigating Neutral Beam Injection
(NBI) power scaling and neutron response in a selected cohort of MAST
experimental discharges.

The project uses Python to profile, process and segment public
experimental data, then presents the derived results in an interactive
Power BI report.

## Project Overview

The analysis focuses on the MAST experiment objective **"Determine power
scaling of NBI fast ion population"**, covering shots **27929--27938**.

The workflow was designed to:

-   extract and profile relevant diagnostic signals;
-   validate signal timebases and data quality;
-   identify stable NBI operating segments;
-   calculate segment-level NBI, neutron and plasma-current statistics;
-   compare SS-only and combined SS + SW NBI operation;
-   assess within-segment signal variability;
-   present the results in Power BI.

## Analysis Workflow

``` text
Public MAST experimental data
            ↓
      HDF5 / NetCDF data
            ↓
      Python extraction
            ↓
 Signal profiling & validation
            ↓
    NBI segment detection
            ↓
     Feature engineering
            ↓
   Processed CSV datasets
            ↓
        Power BI
            ↓
 Scientific interpretation
```

## Dashboard

The Power BI report contains four analysis pages.

### 01 --- Overview

Cohort-level overview of NBI power and neutron response, including:

-   shots and NBI segments analysed;
-   peak NBI power;
-   peak neutron rate;
-   neutron rate versus NBI power;
-   integrated neutron output by shot;
-   SS-only and SS + SW configuration comparison.

### 02 --- Segment Response

Segment-level analysis examining:

-   mean NBI power;
-   mean neutron rate;
-   mean plasma current;
-   plasma current versus NBI power;
-   neutron response by NBI segment;
-   detailed segment operating conditions.

### 03 --- Shot Detail

Interactive single-shot investigation showing:

-   number of NBI segments;
-   peak NBI power;
-   peak neutron rate;
-   mean plasma current;
-   NBI power by segment;
-   neutron response by segment;
-   segment timing and operating conditions.

### 04 --- Uncertainty & Data Quality

Assessment of within-segment signal variability using:

-   standard deviation of NBI power;
-   standard deviation of neutron rate;
-   standard deviation of plasma current;
-   coefficient of variation (CV) for relative signal variability.

> **Note:** Standard deviation and CV on this page describe
> within-segment signal variability. They should not be interpreted as a
> complete metrological measurement-uncertainty budget.

## Data Source

Raw experimental data are **not redistributed in this repository**.

The source data are publicly available through the **UKAEA Open Data ---
MAST Data** archive:

https://opendata.ukaea.uk/mast-data/

The archive identifies shots **27929--27938** as the experiment:

**Determine power scaling of NBI fast ion population**

Users wishing to reproduce the analysis should download the required
diagnostic files for the relevant shots from the original UKAEA source.

## Repository Structure

``` text
mast-u-nbi-power-scaling/
│
├── README.md
├── src/
│   └── Python analysis scripts
├── data/
│   └── Processed CSV outputs
├── dashboard/
│   └── Power BI dashboard PDF
└── images/
    └── Dashboard screenshots
```

## Processed Data

The repository can include the final processed datasets used by Power BI
rather than copies of the raw experimental files.

Important derived variables include:

-   shot number;
-   NBI segment number;
-   NBI configuration;
-   segment start and end time;
-   segment duration;
-   mean NBI power;
-   mean neutron rate;
-   mean plasma current;
-   within-segment standard deviations;
-   segment-to-segment change metrics.

## Tools

-   **Python**
-   **pandas**
-   **NumPy**
-   **h5py**
-   **Power BI**
-   **DAX**
-   **Git / GitHub**

## Reproducing the Analysis

1.  Download the required MAST shot data from the UKAEA Open Data
    archive.
2.  Place the source files in a local working directory.
3.  Run the Python scripts in `src/` in the documented processing order.
4.  Generate the processed segment-level datasets.
5.  Load the processed outputs into Power BI.
6.  Recreate or inspect the dashboard analysis.

Raw experimental files are intentionally excluded from the repository.

## Key Analytical Themes

This project explores:

1.  How neutron response varies with NBI power.
2.  Differences between SS-only and combined SS + SW operation.
3.  Segment-level plasma behaviour.
4.  Changes within individual shots.
5.  Within-segment signal stability and relative variability.

## Limitations

-   The analysis covers a small, targeted experimental cohort.
-   Not every source shot necessarily contributes a valid final NBI
    segment.
-   Segment identification depends on the processing and stability
    criteria implemented in the Python pipeline.
-   Relationships shown in the dashboard should not automatically be
    interpreted as causal.
-   Standard deviation and coefficient of variation quantify observed
    signal variability and are not, by themselves, complete
    measurement-uncertainty estimates.

## Dashboard Export

A static PDF export of the final Power BI report is provided in the
`dashboard/` folder.

For the best experience, use the published interactive Power BI report
where access is available.

## Author

Portfolio project demonstrating scientific data processing, Python
analysis, feature engineering and Power BI visualisation using public
fusion experimental data.
